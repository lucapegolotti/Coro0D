from parse import *
from connectivity import *
from plot_tools import *

def main():
    pathsfdr = "/Users/luca/HICAData/Patient_1/CoronaryP1/"
    paths = parse_vessels(pathsfdr)
    # plot_vessel_portions(paths)
    chunks = build_slices(paths)

if __name__ == "__main__":
    main()
