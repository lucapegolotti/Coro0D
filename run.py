from parse import *
from connectivity import *
from plot_tools import *
from physical_block import *

def main():
    pathsfdr = "examples/CoronaryP1/"
    paths = parse_vessels(pathsfdr)
    chunks, bifurcations, connectivity = build_slices(paths)
    blocks = create_physical_blocks(chunks, model_type = 'Windkessel2')
    plot_vessel_portions(chunks, bifurcations, connectivity)

if __name__ == "__main__":
    main()
