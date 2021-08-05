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
        self.tol = 0.45
        # maxlength of the single vessel portion
        self.maxlength = 4 * self.tol
        # density of blood
        self.density = 1.06
        # viscosity of blood
        self.viscosity = 0.04
        # elastic modulus
        # taken from "Measurement of the uniaxial mechanical properties of healthy
        # and atherosclerotic human coronary arteries"
        self.E = 1.5 * 10 ** 7
        # vessel thickness ratio w.r.t. diameter
        self.thickness_ratio = 0.08
        # use pressure at inlet
        self.use_inlet_pressure = True
        # timestep size
        self.deltat = 0.00025
        # initial time
        self.t0 = 0.0
        # final time
        self.T = 3.0
        # ramp rime
        self.t0ramp = -0.3
        # index of the first minima to be considered
        self.starting_minima = 0
        # self length units of the geometry files
        self.units = "cm"
        # coronary side
        self.side = "right"
        # run an healthy simulation (i.e. no stenotic branches)
        self.isHealthy = False
        # name of the inlet branch
        self.inlet_name = 'RCA'
        # array of positions of the stenoses
        self.stenoses = dict()
        self.stenoses['RCA'] = [42, 43, 44, 45, 46]
        # threshold of the metric to automatically detect stenoses
        self.threshold_metric = 0.80
        # minimal stenosis length
        self.min_stenoses_length = 0.80
        # use automatic stenoses detection
        self.autodetect_stenoses = True


class SolverData:
    def __init__(self):
        # tolerance on the relative error in Newton's iterations
        self.tol = 1e-6
        # minimal absolute error in Newton's iterations
        self.min_err = 1e-15
        # maximal number of Newton's iterations
        self.max_iter = 100
        # treatment of the non-linear term
        self.strategy = "semi-implicit"

        return


def main():
    pd = ProblemData()
    sd = SolverData()
    fdr = os.getcwd()
    paths = parse_vessels(fdr, pd)
    chunks, bifurcations, connectivity = build_slices(paths, pd)
    plot_vessel_portions(chunks, bifurcations, connectivity, color="stenosis")
    show_stenoses_details(chunks, pd.tol)

    stenotic_portions = [23] if not pd.isHealthy else [25]  # TO BE SET MANUALLY

    coeff_resistance = 0.81
    coeff_capacitance = 0.55
    rc = RCCalculator(fdr, pd.side, coeff_resistance, coeff_capacitance)
    rc.assign_resistances_to_outlets(chunks, connectivity)
    rc.assign_capacitances_to_outlets(chunks, connectivity)
    rc.assign_downstream_resistances(chunks, connectivity)

    bcmanager = BCManager(chunks, connectivity,
                          inletbc_type="pressure",
                          outletbc_type="coronary",
                          folder=fdr,
                          problem_data=pd,
                          distal_pressure_coeff=0.75,
                          distal_pressure_shift=15)

    # setting up the blocks and computing a steady solution
    blocks = create_physical_blocks(chunks, model_type='Windkessel2', stenosis_model_type='YoungTsai',
                                    problem_data=pd, folder=fdr, connectivity=connectivity)

    ode_system_steady = ODESystem(blocks, connectivity, bcmanager)
    sol_steady = ode_system_steady.solve_steady()
    for stenotic_portion in stenotic_portions:
        print(f"Steady flow in portion 23: {sol_steady[stenotic_portion * 3 + 2]}\n")

    # re-setting up the blocks, using the pre-computed steady solution, and solving the unsteady problem
    blocks = create_physical_blocks(chunks, model_type='Windkessel2', stenosis_model_type='ItuSharma',
                                    problem_data=pd, folder=fdr, connectivity=connectivity, sol_steady=sol_steady)
    ode_system = ODESystem(blocks, connectivity, bcmanager)
    tma = BDF2(ode_system, connectivity, pd, sd, bcmanager)
    solutions, times = tma.run()

    show_inlet_flow_vs_pressure(solutions, times, bcmanager, pd.t0, pd.T)
    # show_animation(solutions, times, pd.t0, chunks, 'Q', resample=4,
    #                inlet_index=bcmanager.inletindex)

    # show_inlet_vs_distal_pressure(bcmanager, pd.t0, pd.T)

    for stenotic_portion in stenotic_portions:
        plot_solution(solutions, times, pd.t0, pd.T, chunks, stenotic_portion, 'Q')
        plot_FFR(solutions, times, pd.t0, pd.T, bcmanager, stenotic_portion, 'Pout')

    positive_times = np.where(times > pd.t0)[0]
    Pin = solutions[bcmanager.inletindex * 3 + 0, positive_times]
    Qin = solutions[bcmanager.inletindex * 3 + 2, positive_times]
    CO = np.loadtxt(os.path.join(fdr, os.path.normpath("Data/cardiac_output.txt")), ndmin=1)[0]
    CO *= (1000 / 60)  # conversion from L/min to mL/s
    CO *= 0.04  # 4% of flow goes in coronaries
    CO *= (0.7 if pd.side == "left" else 0.3 if pd.side == "right" else 0.0)
    print("\nFlow = " + str(simps(Qin, times[positive_times]) / (pd.T - pd.t0)) + " [mL/s]")
    print("Target Flow = " + str(CO) + " [mL/s]")
    print("Mean inlet pressure = " + str(simps(Pin, times[positive_times]) / 1333.2 / (pd.T - pd.t0)) + " [mmHg]")

    ow = OutputWriter("Output", bcmanager, chunks, pd)
    ow.write_outlet_rc()
    npoints = 10001
    ow.write_distal_pressure(pd, npoints)
    ow.write_inlet_pressure(pd, npoints)
    ow.write_inlet_outlets_flow_pressures(times, solutions, chunks, bcmanager)
    ow.write_thickess_caps(paths)

    plot_show()

    return


if __name__ == "__main__":
    main()
