import sys
import os
sys.path.append(os.path.normpath("../../core/"))

from core.parse import *
from core.connectivity import *
from core.plot_tools import *
from core.physical_block import *
from core.ode_system import ODESystem
from core.bdf import BDF2
from core.bcmanager import BCManager
from core.rc_calculator import RCCalculator
from scipy.integrate import simps
from core.output_writer import OutputWriter


class ProblemData:
    def __init__(self):
        # tolerance to determine if two points are the same
        self.tol = 0.4
        # maxlength of the single vessel portion
        self.maxlength = 5 * self.tol
        # density of blood
        self.density = 1.06
        # viscosity of blood
        self.viscosity = 0.04
        # elastic modulus
        self.E = 2 * 10 ** 5
        # vessel thickness ratio w.r.t. diameter
        self.thickness_ratio = 0.08
        # use pressure at inlet
        self.use_inlet_pressure = True
        # timestep size
        self.deltat = 0.005
        # initial time
        self.t0 = 0.0
        # final time
        self.T = 8
        # ramp rime
        self.t0ramp = -0.3
        # index of the first minima to be considered
        self.starting_minima = 1
        # self length units of the geometry files
        self.units = "cm"


def main():
    pd = ProblemData()
    coronary = "left"
    fdr = os.getcwd()
    paths = parse_vessels(fdr, pd)
    chunks, bifurcations, connectivity = build_slices(paths, pd.tol, pd.maxlength)
    plot_vessel_portions(chunks, bifurcations, connectivity)

    coeff_resistance = 0.995
    coeff_capacitance = 0.6
    rc = RCCalculator(fdr, coronary, coeff_resistance, coeff_capacitance)
    rc.assign_resistances_to_outlets(chunks, connectivity)
    rc.assign_capacitances_to_outlets(chunks, connectivity)

    blocks = create_physical_blocks(chunks, model_type='Windkessel2', problem_data=pd)
    bcmanager = BCManager(chunks, connectivity,
                          inletbc_type="pressure",
                          outletbc_type="coronary",
                          folder=fdr,
                          problem_data=pd,
                          coronary=coronary,
                          distal_pressure_coeff=1.05,
                          distal_pressure_shift=10)

    ode_system = ODESystem(blocks, connectivity, bcmanager)
    bdf = BDF2(ode_system, connectivity, pd, bcmanager)
    solutions, times = bdf.run()

    show_inlet_flow_vs_pressure(solutions, times, bcmanager, 0, 2)
    show_animation(solutions, times, pd.t0, chunks, 'Q', resample=4,
                   inlet_index=bcmanager.inletindex)

    plot_solution(solutions, times, pd.t0, pd.T, chunks, 8, 'Q')
    show_inlet_vs_distal_pressure(bcmanager, 0, 1)

    positive_times = np.where(times > pd.t0)[0]
    Pin = solutions[bcmanager.inletindex * 3 + 0, positive_times]
    Qin = solutions[bcmanager.inletindex * 3 + 2, positive_times]
    print("Flow = " + str(simps(Qin, times[positive_times]) / (pd.T - pd.t0)) + " [mL/s]")
    print("Mean inlet pressure = " + str(simps(Pin, times[positive_times]) / 1333.2 / (pd.T - pd.t0)) + " [mmHg]")

    ow = OutputWriter("output", bcmanager, chunks, pd)
    ow.write_outlet_rc()
    npoints = 10001
    ow.write_distal_pressure(pd, npoints)
    ow.write_inlet_pressure(pd, npoints)

    plot_show()

    return


if __name__ == "__main__":
    main()