from parse import *
from connectivity import *
from plot_tools import *
from physical_block import *
from ode_system import ODESystem
from bdf import BDF1

def main():
    pathsfdr = "examples/CoronaryP1/"
    paths = parse_vessels(pathsfdr)
    chunks, bifurcations, connectivity = build_slices(paths)
    blocks = create_physical_blocks(chunks, model_type = 'Windkessel2')
    ode_system = ODESystem(blocks, connectivity)
    bdf = BDF1(ode_system, connectivity)
    plot_vessel_portions(chunks, bifurcations, connectivity)

if __name__ == "__main__":
    main()
