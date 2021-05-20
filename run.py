from parse import *
from connectivity import *
from plot_tools import *
from physical_block import *
from ode_system import ODESystem
from bdf import BDF1
from problem_data import ProblemData
from bcmanager import BCManager
import matplotlib.pyplot as plt

def main():
    pd = ProblemData()
    fdr = "examples/CoronaryP1/"
    paths = parse_vessels(fdr)
    chunks, bifurcations, connectivity = build_slices(paths, pd.tol, pd.maxlength)
    blocks = create_physical_blocks(chunks, model_type = 'Windkessel2', problem_data = pd)
    bcmanager = BCManager(chunks, connectivity,
                          inletbc_type = "pressure",
                          outletbc_type = "resistance",
                          folder = fdr,
                          problem_data = pd)
    ode_system = ODESystem(blocks, connectivity, bcmanager)
    bdf = BDF1(ode_system, connectivity, pd, bcmanager)
    # plot_vessel_portions(chunks, bifurcations, connectivity)
    solutions, times = bdf.run()
    # fig, ax1, ax2 = plot_solution(solutions, times, pd.t0, pd.T, chunks, 3, 'Pout')
    # ax2.plot(bcmanager.inletbc.times,
    #          np.array(bcmanager.inletbc.pressure_values) / 1333.2,
    #          color = 'red',
    #          linestyle='dashed')
    show_animation(solutions, times, chunks, 'Pin', resample = 5)

if __name__ == "__main__":
    main()
