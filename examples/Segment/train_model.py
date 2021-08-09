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
    plot_vessel_portions(chunks, bifurcations, connectivity)
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
    def __init__(self, inputdim, stencil):
        super().__init__()

        self.dim = inputdim
        self.stencil = stencil
        self.selector = np.zeros(shape = (self.stencil, self.dim))
        for i in range(0, self.stencil):
            self.selector[i,i] = -1

        self.selector = tf.convert_to_tensor(self.selector, dtype=tf.float32)

        self.expander = np.zeros(shape = (self.stencil, self.stencil - 1))

        half = int(np.floor((self.stencil - 1) / 2))
        # top half
        for i in range(0, half):
            self.expander[i,i] = 1

        # bottom half
        for i in range(half, self.stencil - 1):
            self.expander[i+1,i] = 1

        self.expander = tf.convert_to_tensor(self.expander, dtype=tf.float32)

        self.one = np.zeros(shape = (self.stencil, 1));
        self.one[half] = 1
        self.one = tf.convert_to_tensor(self.one, dtype=tf.float32)

    def call(self, inputs):
        A = tf.matmul(self.selector, inputs[0], transpose_b=True)
        B = tf.matmul(self.expander, inputs[1], transpose_b=True)
        C = tf.add(B, self.one)
        res = tf.matmul(A, C, transpose_a=True)
        # print(A)
        # print(inputs[0])
        return res

def main():
    solutions, times, chunks = run0D()

    stencil = 5
    ghosts = int((stencil - 1)/2)

    # extract all the pressures
    ndomains = len(chunks)

    ps = solutions[0::3,:]
    ps = ps[0:ndomains]
    ps = np.vstack((ps,solutions[3 * ndomains + 1,:]))

    size1 = ps.shape[0]
    size2 = ps.shape[1]

    # pad pressure
    pad = np.zeros((ghosts, size2))
    ps = np.vstack((pad,ps,pad))

    radii = np.zeros(shape=(0))
    coords = []

    # add intial ghost nodes
    for i in range(0, ghosts):
        coords.append([0,0,0])
        radii = np.hstack((radii, 0))

    for chunk in chunks:
        coords.append(chunk.coords[0,:])
        # posindices = np.where(chunk.radii > 0)
        radii = np.hstack((radii, chunk.radii[0]))

    # append also the end point
    coords.append(chunks[-1].coords[-1,:])
    radii = np.hstack((radii, chunks[-1].radii[-1]))

    # add final ghost nodes
    for i in range(0, ghosts):
        coords.append([0,0,0])
        radii = np.hstack((radii, 0))

    # build dataset. We sweep the nodes and add each stencil + size of the
    # neighboring elements
    inindex = ghosts
    finindex = ghosts + ndomains
    X = []
    for i in range(inindex, finindex + 1):
        hs = []
        rs = []
        for j in range(i-ghosts, i+ghosts):
            rs.append(radii[j])
            if (np.linalg.norm(coords[j]) < 1e-16 or
                np.linalg.norm(coords[j + 1]) < 1e-16):
                hs.append(0)
            else:
                hs.append(np.linalg.norm(coords[j+1] - coords[j]))
        rs.append(radii[i+ghosts])

        # we skip the first timestep because we don't have an initial condition
        for jtime in range(1,size2):
            x = ps[i-ghosts:i+ghosts+1,jtime]
            x = np.hstack((x,ps[i-ghosts:i+ghosts+1,jtime-1]))
            x = np.hstack((x,np.array(hs)))
            x = np.hstack((x,np.array(rs)))
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

    inputdim = stencil * 4 - 1
    inputs = tf.keras.Input(shape=(stencil * 4 - 1,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(stencil-1)(x)
    outputs = mul(inputdim, stencil)((inputs, x))

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
              metrics=['mae'])

    model.fit(X, Y, epochs=5, batch_size=1, validation_split=0.2)
    model.save('saved_model/')

if __name__ == "__main__":
    main()
