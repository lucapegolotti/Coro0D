from parse import *
from connectivity import *
from plot_tools import *

def main():
    pathsfdr = "examples/CoronaryP1/"
    paths = parse_vessels(pathsfdr)
    chunks, bifurcations, connectivity = build_slices(paths)
    plot_vessel_portions(chunks, bifurcations, connectivity)

if __name__ == "__main__":
    main()
