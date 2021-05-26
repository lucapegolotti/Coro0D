import sys

# add path to core
sys.path.append("../../core/")

from parse import *
from connectivity import *
from plot_tools import *
from physical_block import *
from ode_system import ODESystem
from bdf import BDF1, BDF2
from problem_data import ProblemData
from bcmanager import BCManager
import matplotlib.pyplot as plt
from rc_calculator import RCCalculator
from scipy.integrate import simps


def main():
    pd = ProblemData()
    coronary = "right"
    fdr = "./"
    paths = parse_vessels(fdr)
    chunks, bifurcations, connectivity = build_slices(paths, pd.tol, pd.maxlength)
    coeff_resistance = 0.8
    coeff_capacitance = 0.3
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
                          distal_pressure_coeff = 0.54,
                          distal_pressure_shift = 15)
    ode_system = ODESystem(blocks, connectivity, bcmanager)
    bdf = BDF2(ode_system, connectivity, pd, bcmanager)
    solutions, times = bdf.run()
    show_inlet_flow_vs_pressure(solutions, times, 5, 6, 9)
    show_animation(solutions, times, pd.t0, chunks, 'Q', resample = 4,
                   inlet_index = bcmanager.inletindex)

    show_inlet_vs_distal_pressure(bcmanager, 0, 1)

    # check total flow / min
    positive_times = np.where(times > pd.t0)[0]
    Pin = solutions[bcmanager.inletindex * 3 + 0, positive_times]
    Qin = solutions[bcmanager.inletindex * 3 + 2, positive_times]
    print("Flow = " + str(simps(Qin, times[positive_times]) / (pd.T - pd.t0)) + " [mL/min]")
    print("Mean inlet pressure = " + str(simps(Pin, times[positive_times]) / 1333.2 / (pd.T - pd.t0)) + " [mmHg]")
    plot_show()

if __name__ == "__main__":
    main()
