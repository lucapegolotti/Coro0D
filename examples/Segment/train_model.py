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

def run0D():
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
    blocks = create_physical_blocks(chunks, model_type = 'Windkessel2', problem_data = pd)
    bcmanager = BCManager(chunks, connectivity,
                          inletbc_type = "pressure",
                          outletbc_type = "zero",
                          folder = fdr,
                          problem_data = pd,
                          coronary = coronary,
                          distal_pressure_coeff = 0.0,
                          distal_pressure_shift = 10)
    ode_system = ODESystem(blocks, connectivity, bcmanager)
    bdf = BDF2(ode_system, connectivity, pd, bcmanager)
    solutions, times = bdf.run()
    # show_animation(solutions, times, pd.t0, chunks, 'Pin', resample = 4,
    #                 inlet_index = bcmanager.inletindex)
    return solutions, times, chunks



class mul(tf.keras.layers.Layer):
    def __init__(self, inputdim, stencil, mins, maxs, center):
        super().__init__()

        self.dim = inputdim
        self.stencil = stencil
        self.selector = np.zeros(shape = (self.stencil + 1, self.dim))
        for i in range(0, self.stencil):
            self.selector[i,i] = 1

        print(self.selector)
        self.selector = tf.convert_to_tensor(self.selector, dtype=tf.float32)

        self.expander = np.zeros(shape = (self.stencil + 1, self.stencil))

        for i in range(0, center):
            self.expander[i,i] = 1

        # bottom half
        for i in range(center, self.stencil):
            self.expander[i+1,i] = 1
        print(self.expander)
        self.expander = tf.convert_to_tensor(self.expander, dtype=tf.float32)

        self.one = np.zeros(shape = (self.stencil + 1, 1));
        self.one[center] = 1
        self.one = tf.convert_to_tensor(self.one, dtype=tf.float32)
        #                                                          this is to have 1 as last component
        self.mins = tf.convert_to_tensor(np.hstack((mins[0:stencil].squeeze(),-1)),
                                         dtype=tf.float32)
        self.maxs = tf.convert_to_tensor(np.hstack((maxs[0:stencil].squeeze(),0)),
                                         dtype=tf.float32)
        print(np.hstack((maxs[0:stencil].squeeze(),0)))
        self.diff = tf.convert_to_tensor(np.hstack(((maxs[0:stencil] - mins[0:stencil]).squeeze(),0)),
                                         dtype=tf.float32)

    def call(self, inputs):
        A = tf.matmul(self.selector, inputs[0], transpose_b=True)
        # rescale velocity to original range
        A = tf.multiply(A, self.diff)
        A = tf.add(A, self.mins)
        B = tf.matmul(self.expander, inputs[1], transpose_b=True)
        C = tf.add(B, self.one)
        res = tf.matmul(A, C, transpose_a=True)
        return res

def generate_datasets(solutions, times, chunks, stencil):
    half = int((stencil - 1)/2)

    # extract all the pressures
    ndomains = len(chunks)

    ps = solutions[0::3,:]
    ps = ps[0:ndomains]
    ps = np.vstack((ps,solutions[3 * ndomains + 1,:]))
    qs = solutions[2::3,:]
    qs = ps[0:ndomains]
    qs = np.vstack((qs,solutions[3 * ndomains + 2,:]))

    nnodes = ps.shape[0]

    size1 = ps.shape[0]
    size2 = ps.shape[1]

    radii = np.zeros(shape=(0))
    coords = []

    for chunk in chunks:
        coords.append(chunk.coords[0,:])
        radii = np.hstack((radii,chunk.radii[0]))

    # append also the end point
    coords.append(chunks[-1].coords[-1,:])
    radii = np.hstack((radii, chunks[-1].radii[-1]))

    Xs = {}
    Ys = {}
    mss = {}
    Mss = {}

    for stsize in range(half+1,stencil+1):
        # build dataset. We sweep the nodes and add each stencil + size of the
        # neighboring elements
        X = []
        for i in range(0, nnodes-stsize-1):
            hs = []
            As = []
            for j in range(i, i+stsize-1):
                As.append(radii[j])
                hs.append(np.linalg.norm(coords[j+1] - coords[j]))
            As.append(radii[i+stsize-1])
            
            # we skip the first timestep because we don't have an initial condition
            for jtime in range(1,size2):
                x = ps[i:i+stsize,jtime]
                x = np.hstack((x,ps[i:i+stsize,jtime-1]))
                x = np.hstack((x,qs[i:i+stsize,jtime]))
                x = np.hstack((x,qs[i:i+stsize,jtime-1]))
                x = np.hstack((x,np.array(hs)))
                x = np.hstack((x,np.array(As)))
                X.append(x)
        X = np.array(X)
        np.random.shuffle(X) # not sure if this is needed

        ms = []
        Ms = []
        # normalization
        for column in range(X.shape[1]):
            m = np.min(X[:,column])
            M = np.max(X[:,column])
            X[:,column] = (X[:,column] - m) / (M - m)
            ms.append(m)
            Ms.append(M)

        Y = np.zeros((X.shape[0],1))
        ms = np.array(ms).reshape(stsize * 6 - 1,1)
        Ms = np.array(Ms).reshape(stsize * 6 - 1,1)
        np.save("dataset/X_st" + str(stsize) + ".npy", X)
        np.save("dataset/Y_st" + str(stsize) + ".npy", Y)
        np.save("dataset/min_st" + str(stsize) + ".npy", ms)
        np.save("dataset/Max_st" + str(stsize) + ".npy", Ms)
        Xs[stsize] = X
        Ys[stsize] = Y
        mss[stsize] = ms
        Mss[stsize] = Ms
    return Xs, Ys, mss, Mss

def main():
    stencil = 5
    half = int((stencil - 1)/2)

    solutions, times, chunks = run0D()
    Xs, Ys, ms, Ms = generate_datasets(solutions, times, chunks, stencil)

    for center in range(2, 3):
        init = np.max((0,center-half))
        end = np.min((stencil-1,center+half))
        shift = 0
        if (end == stencil-1):
            shift = center+half - end

        stsize = end - init + 1
        inputdim = stsize * 6 - 1
        inputs = tf.keras.Input(shape=(inputdim,))
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(stsize, name='coeffs')(x)
        outputs = mul(inputdim, stsize, ms[stsize], Ms[stsize], center - shift)((inputs, x))

        model = tf.keras.Model(inputs = inputs, outputs = outputs)

        model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
                  metrics=['mae'])
        #         run_eagerly=None)

        model.fit(Xs[stsize], Ys[stsize], epochs=30, batch_size=1, validation_split=0.1)
        model.save('saved_model/center' + str(center))

if __name__ == "__main__":
    main()
