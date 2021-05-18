from parse import *
from connectivity import *
from plot_tools import *
from physical_block import *
from ode_system import ODESystem

def main():
    pathsfdr = "examples/CoronaryP1/"
    paths = parse_vessels(pathsfdr)
    chunks, bifurcations, connectivity = build_slices(paths)
    blocks = create_physical_blocks(chunks, model_type = 'Windkessel2')
    ode_system = ODESystem(blocks, connectivity)
    ode_system.get_system_matrix()
    plot_vessel_portions(chunks, bifurcations, connectivity)

if __name__ == "__main__":
    main()
