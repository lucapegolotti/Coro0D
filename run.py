from parse import *
from connectivity import *
from plot_tools import *
from physical_block import *
from ode_system import ODESystem
from bdf import BDF1
from problem_data import ProblemData
from bcmanager import BCManager

def main():
    pd = ProblemData()
    pathsfdr = "examples/CoronaryP1/"
    paths = parse_vessels(pathsfdr)
    chunks, bifurcations, connectivity = build_slices(paths, pd.tol, pd.maxlength)
    blocks = create_physical_blocks(chunks, model_type = 'Windkessel2', problem_data = pd)
    bcmanager = BCManager(chunks, connectivity, \
                          inletbc_type = "pressure", \
                          outletbc_type = "resistance")
    ode_system = ODESystem(blocks, connectivity, bcmanager)
    bdf = BDF1(ode_system, connectivity, pd, bcmanager)
    plot_vessel_portions(chunks, bifurcations, connectivity)
    solutions, times = bdf.run()
    plot_solution(solutions, times, chunks, 3, 'Pin')

if __name__ == "__main__":
    main()
