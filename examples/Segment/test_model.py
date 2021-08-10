import sys
import os

# add path to core
sys.path.append("../../core/")

from parse import *
from connectivity import *
from plot_tools import *
from physical_block import *
from ode_system import ODESystem
from bdf import BDF1, BDF2
from bcmanager import BCManager
import matplotlib.pyplot as plt
from rc_calculator import RCCalculator
from scipy.integrate import simps
from output_writer import OutputWriter
import tensorflow as tf
import math
from keras.models import Model

class ProblemData:
    def __init__(self):
        # tolerance to determine if two points are the same
        self.tol = 0.04
        # maxlength of the singe vessel portion
        self.maxlength = 5 * self.tol
        # density of blood
        self.density = 1.06
        # viscosity of blood
        self.viscosity = 0.04
        # elastic modulus
        # taken from "Measurement of the uniaxial mechanical properties of healthy
        # and atherosclerotic human coronary arteries"
        self.E = 1.5 * 10**7
        # vessel thickness ration w.r.t. diameter
        self.thickness_ratio = 0.08
        # use pressure at inlet
        self.use_inlet_pressure = True
        # timestep size
        self.deltat = 0.01
        # initial time
        self.t0 = 0.0
        # final time
        self.T = 10
        # ramp rime
        self.t0ramp = -0.3
        # self length units of the geometry files
        self.units = "cm"

def create_structures():
    pd = ProblemData()
    coronary = "right"
    fdr = "./"
    paths = parse_vessels(fdr, pd)
    chunks, bifurcations, connectivity = build_slices(paths, pd.tol, pd.maxlength)
    # plot_vessel_portions(chunks, bifurcations, connectivity)
    coeff_resistance = 1.0
    coeff_capacitance = 1.0
    rc = RCCalculator(fdr, coronary, coeff_resistance, coeff_capacitance)
    rc.assign_resistances_to_outlets(chunks, connectivity)
    rc.assign_capacitances_to_outlets(chunks, connectivity)
    bcmanager = BCManager(chunks, connectivity,
                          inletbc_type = "pressure",
                          outletbc_type = "zero",
                          folder = fdr,
                          problem_data = pd,
                          coronary = coronary,
                          distal_pressure_coeff = 0.0,
                          distal_pressure_shift = 10)

    return chunks, bifurcations, connectivity, bcmanager

def load_dataset():
    X = np.load('dataset/X.npy')
    Y = np.load('dataset/Y.npy')
    ms = np.load('dataset/min.npy')
    Ms = np.load('dataset/Max.npy')

    # rescaled X
    rX = np.copy(X)
    for i in range(0, X.shape[1]):
        rX[:,i] = ms[i] + rX[:,i] * (Ms[i] - ms[i])

    return X, Y, ms, Ms, rX

def load_model():
    original_model = tf.keras.models.load_model('saved_model/')

    layer_name = 'coeffs'
    model = Model(inputs=original_model.input,
                       outputs=original_model.get_layer(layer_name).output)


    # maybe check why this is not exactly the same
    # ind = 10
    # coeffs = coeffmodel(np.expand_dims(X[ind,:], axis=0)).numpy().squeeze()
    # a = coeffs[0] * rX[ind,0] + coeffs[1] * rX[ind,1] + rX[ind,2] + coeffs[2] * rX[ind,3] + coeffs[3] * rX[ind,4]
    # print(a)
    # print(model(np.expand_dims(X[ind,:], axis=0)))

    return model

def create_geometrical_vectors(chunks, stencil):
    # ghosts = int((stencil - 1)/2)
    #
    # # add intial ghost nodes
    # for i in range(0, ghosts):
    #     coords.append([0,0,0])
    #     radii = np.hstack((radii, 0))
    #
    # for chunk in chunks:
    #     coords.append(chunk.coords[0,:])
    #     radii = np.hstack((radii, -1))
    #
    # # append also the end point
    # coords.append(chunks[-1].coords[-1,:])
    # radii = np.hstack((radii, chunks[-1].radii[-1]))
    #
    # # add final ghost nodes
    # for i in range(0, ghosts):
    #     coords.append([0,0,0])
    #     radii = np.hstack((radii, 0))
    nchunks = len(chunks)
    gvecs = []

    half = int(np.floor((stencil - 1)*0.5))
    for ic in range(0, nchunks + 1):
        coords = []
        radii = []
        cgvec = []
        for jc in range(ic - half, ic + half + 1):
            if jc < 0 or jc >= nchunks + 1:
                coords.append(np.array([0,0,0]))
                radii.append(0)
            elif jc < nchunks:
                coords.append(np.array(chunks[jc].coords[0,:]))
                radii.append(float(chunks[jc].radii[0,:]))
            elif jc == nchunks:
                coords.append(np.array(chunks[jc-1].coords[-1,:]))
                radii.append(float(chunks[jc-1].radii[-1,:]))

        for jr in range(0, stencil - 1):
            if (np.linalg.norm(coords[jr+1]) > 0 and
                np.linalg.norm(coords[jr]) > 0):
                cgvec.append(np.linalg.norm(coords[jr+1] - coords[jr]))
            else:
                cgvec.append(0)
        for radius in radii:
            cgvec.append(radius**2 * math.pi)
        gvecs.append(np.array(cgvec))

    return gvecs

def assemble_matrix(sol, pastsol, gvecs, stencil, model, mins, maxs):
    N = sol.shape[0]
    A = np.zeros((N,N))

    half = int(np.floor((stencil - 1)*0.5))

    zeros = np.zeros((half,))
    # pad sol and pastsol
    sol = np.hstack((zeros,sol,zeros))
    pastsol = np.hstack((zeros,pastsol,zeros))
    for i in range(half, N + half):
        actuali = i - half
        x = np.hstack((sol[i-half:i+half+1],
                       pastsol[i-half:i+half+1],
                       gvecs[actuali]))
        x = np.divide(x - mins,maxs - mins)
        coefs = model(x.reshape((1,x.size))).numpy().squeeze()
        count = 0
        for j in range(actuali - half, actuali + half + 1):
            if j < 0 or j >= N:
                count = count + 1
                continue
            if j >= 0 and j < actuali:
                A[actuali, j] = coefs[count]
            elif j == actuali:
                A[actuali, j] = 1
            elif j < N and j > actuali:
                A[actuali, j] = coefs[count-1]
            count = count + 1

    return A

def compute_jacobian(sol, pastsol, gvecs, stencil, model, mins, maxs, A):
    # return A
    N = sol.shape[0]
    J = np.zeros((N,N))

    half = int(np.floor((stencil - 1)*0.5))

    zeros = np.zeros((half,))
    # pad sol and pastsol
    sol = np.hstack((zeros,sol,zeros))
    pastsol = np.hstack((zeros,pastsol,zeros))
    for i in range(half, N + half):
        actuali = i - half
        x = np.hstack((sol[i-half:i+half+1],
                       pastsol[i-half:i+half+1],
                       gvecs[actuali]))
        x = np.divide(x - mins,maxs - mins)
        xt = tf.convert_to_tensor(x.reshape((1,x.size)))
        with tf.GradientTape() as g:
            g.watch(xt)
            y = model(xt)
        grads = g.jacobian(y, xt).numpy().squeeze()[:,0:stencil]
        jacbs = np.matmul(grads,x[0:stencil])

        count = 0
        for j in range(actuali - half, actuali + half + 1):
            if j < 0 or j >= N:
                count = count + 1
                continue
            if j >= 0 and j < actuali:
                J[actuali, j] = jacbs[count]
            elif j == actuali:
                J[actuali, j] = 0
            elif j < N and j > actuali:
                J[actuali, j] = jacbs[count-1]
            count = count + 1

    return J + A

def run0D(chunks, bifurcations, connectivity, bcmanager):
    pd = ProblemData()
    blocks = create_physical_blocks(chunks, model_type = 'Windkessel2', problem_data = pd)
    ode_system = ODESystem(blocks, connectivity, bcmanager)
    bdf = BDF2(ode_system, connectivity, pd, bcmanager)
    solutions, times = bdf.run()
    # show_animation(solutions, times, pd.t0, chunks, 'Pin', resample = 4,
    #                 inlet_index = bcmanager.inletindex)
    return solutions, times, chunks


def main():
    pd = ProblemData()
    stencil = 5
    chunks, bifurcations, connectivity, bcmanager = create_structures()
    exsol1, times, chunks = run0D(chunks, bifurcations, connectivity, bcmanager)
    X, Y, ms, Ms, rX = load_dataset()
    ms = ms.squeeze()
    Ms = Ms.squeeze()
    model = load_model()
    gvecs = create_geometrical_vectors(chunks, stencil)
    nchunks = len(chunks)
    exsol = exsol1[0::3,:]
    exsol = exsol1[0:nchunks,:]
    exsol = np.vstack((exsol,exsol1[3 * nchunks + 1,:]))

    solutions = np.zeros((nchunks+1,1))

    t = pd.t0ramp
    dt = pd.deltat
    T = pd.T

    cur = 0
    while t < T:
        print('Solving t = ' + "{:.2f}".format(t) + " s")
        t = t + dt

        maxit = 30
        tol = 1e-6

        err = tol + 1
        it = 0

        rhs = np.zeros((nchunks+1,1))
        sol = solutions[:,cur]
        while it < maxit:
            it = it + 1
            A = assemble_matrix(sol, solutions[:,cur], gvecs, stencil, model, ms, Ms)
            A[0,:] = A[0,:] * 0
            A[0,0] = 1
            A[-1,:] = A[-1,:] * 0
            A[-1,-1] = 1
            bcmanager.inletbc.apply_bc_vector(rhs, t, 0)
            res = np.matmul(A,np.expand_dims(sol,axis=1)) - rhs
            err = np.linalg.norm(res)
            print('\terr = ' + str(err))
            if (err < tol):
                break

            J = compute_jacobian(sol, solutions[:,cur], gvecs, stencil, model, ms, Ms, A)

            dsol = np.linalg.solve(J,res)
            sol = (np.expand_dims(sol,axis=1) - dsol).squeeze()
        print("dnn")
        print(sol)
        print("exact")
        print(exsol[:,cur+1])
        print(sol.shape)
        print(exsol.shape)
        print(A[1,:])
        print(A[2,:])
        print(A[3,:])
        # A = assemble_matrix(exsol[], solutions[:,cur], gvecs, stencil, model, ms, Ms)
        solutions = np.hstack((solutions,np.expand_dims(sol,axis=1)))
        cur = cur + 1





if __name__ == "__main__":
    main()
