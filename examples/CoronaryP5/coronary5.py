import sys

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


class ProblemData:
    def __init__(self):
        # tolerance to determine if two points are the same
        self.tol = 0.4
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
        self.deltat = 0.005
        # initial time
        self.t0 = 0.0
        # final time
        self.T = 10
        # ramp rime
        self.t0ramp = -0.3
        # self length units of the geometry files
        self.units = "cm"


def main():
    pd = ProblemData()
    coronary = "right"
    fdr = "./"
    paths = parse_vessels(fdr, pd)
    chunks, bifurcations, connectivity = build_slices(paths, pd.tol, pd.maxlength)
    plot_vessel_portions(chunks, bifurcations, connectivity)
    coeff_resistance = 0.98
    coeff_capacitance = 0.6
    rc = RCCalculator(fdr, coronary, coeff_resistance, coeff_capacitance)
    rc.assign_resistances_to_outlets(chunks, connectivity)
    rc.assign_capacitances_to_outlets(chunks, connectivity)
    blocks = create_physical_blocks(chunks, model_type = 'Windkessel2', problem_data = pd)
    bcmanager = BCManager(chunks, connectivity,
                          inletbc_type = "pressure",
                          outletbc_type = "coronary",
                          folder = fdr,
                          problem_data = pd,
                          coronary = coronary,
                          distal_pressure_coeff = 1.05,
                          distal_pressure_shift = 10)
    ode_system = ODESystem(blocks, connectivity, bcmanager)
    bdf = BDF2(ode_system, connectivity, pd, bcmanager)
    solutions, times = bdf.run()
    show_inlet_flow_vs_pressure(solutions, times, bcmanager, 0, 2)
    show_animation(solutions, times, pd.t0, chunks, 'Q', resample = 4,
                   inlet_index = bcmanager.inletindex)

    plot_solution(solutions, times, pd.t0, pd.T, chunks, 0, 'Q')
    show_inlet_vs_distal_pressure(bcmanager, 0, 1)

    # check total flow / min
    positive_times = np.where(times > pd.t0)[0]
    Pin = solutions[bcmanager.inletindex * 3 + 0, positive_times]
    Qin = solutions[bcmanager.inletindex * 3 + 2, positive_times]
    print("Flow = " + str(simps(Qin, times[positive_times]) / (pd.T - pd.t0)) + " [mL/min]")
    print("Mean inlet pressure = " + str(simps(Pin, times[positive_times]) / 1333.2 / (pd.T - pd.t0)) + " [mmHg]")
    ow = OutputWriter("output", bcmanager, chunks, pd)
    ow.write_outlet_rc()
    npoints = 101
    ow.write_distal_pressure(pd, npoints)
    ow.write_inlet_pressure(pd, npoints)
    ow.write_inlet_outlets_flow_pressures(times, solutions, chunks, bcmanager)
    ow.write_thickess_caps(paths)
    plot_show()

if __name__ == "__main__":
    main()
