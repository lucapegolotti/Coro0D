import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from vessel_portion import VesselPortion
from contour import Contour


def parse_vessels(fdr, problem_data):
    if problem_data.units == "mm":
        coeff = 0.1
    elif problem_data.units == "cm":
        coeff = 1
    else:
        raise ValueError(problem_data.units + " units not implemented")

    fullpaths = []
    index = 0
    for filename in os.listdir(os.path.join(fdr, "Paths")):
        if filename[0] != ".":
            index = index + 1
            print(filename)
            path = parse_single_path(os.path.join(fdr, "Paths", filename), filename[:-4],
                                     coeff, problem_data.inlet_name)
            if path is not None:
                filenamectgr = filename[0:-4] + ".ctgr"
                path = parse_single_segmentation(os.path.join(fdr, "Segmentations", filenamectgr), path,
                                                 coeff, problem_data.inlet_name)
                if path is not None:
                    fullpaths.append(path)

    return fullpaths


def open_xml(namefile):
    with open(namefile) as f:
        xml = f.read()

    # Add fake root to avoid problems due to presence of multiple roots
    tree = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) +
                         "</root>")

    return tree


def parse_single_path(namefile, pathname, coeff, inlet_name):

    if namefile[-4:] != ".pth":
        print(f"Invalid filename {namefile} in Paths/ folder! Ignoring it while parsing paths.")
        return

    tree = open_xml(namefile)
    path_points = tree[-1][0][0][1]

    if os.path.split(namefile[:-4])[-1] == inlet_name:
        isReversed = (float(path_points[0][0].attrib['z']) - float(path_points[-1][0].attrib['z'])) < 0
    else:
        isReversed = len(tree) == 1

    if isReversed:
        print(f"{namefile} is a reversed path!")

    xs = []
    ys = []
    zs = []

    for child in path_points:
        x = float(child[0].attrib['x']) * coeff
        y = float(child[0].attrib['y']) * coeff
        z = float(child[0].attrib['z']) * coeff

        if not isReversed:
            xs.append(x)
            ys.append(y)
            zs.append(z)
        else:
            xs.insert(0, x)
            ys.insert(0, y)
            zs.insert(0, z)

    single_path = VesselPortion(xs, ys, zs, pathname)
    return single_path


def parse_single_segmentation(namefile, vessel, coeff, inlet_name):

    if namefile[-5:] != ".ctgr":
        print(f"Invalid filename {namefile} in Paths/ folder! Ignoring it while parsing segmentations.")
        return

    tree = open_xml(namefile)

    if os.path.split(namefile[:-4])[-1] == inlet_name:
        isReversed = (float(tree[1][0][1][0][0].attrib['z']) - float(tree[1][0][-1][0][0].attrib['z'])) < 0
    else:
        isReversed = len(tree) == 0

    if isReversed:
        print(f"{namefile} is a reversed path!")

    contours = []
    # we skip the first index because it corresponds to "lofting_parameters"
    ncontours = len(tree[1][0]) - 1

    for icont in range(ncontours):
        curcontour = tree[1][0][icont + 1]
        # we want to use cgs system
        x = float(curcontour[0][0].attrib['x']) * coeff
        y = float(curcontour[0][0].attrib['y']) * coeff
        z = float(curcontour[0][0].attrib['z']) * coeff
        id = int(curcontour[0].attrib['id'])
        control_point = np.array([x, y, z])
        curpoints = curcontour[2]
        ncurpoints = len(curpoints)
        contour = np.zeros([ncurpoints, 3])
        for ipoint in range(ncurpoints):
            contour[ipoint, 0] = float(curpoints[ipoint].attrib['x']) * coeff
            contour[ipoint, 1] = float(curpoints[ipoint].attrib['y']) * coeff
            contour[ipoint, 2] = float(curpoints[ipoint].attrib['z']) * coeff

        if not isReversed:
            contours.append(Contour(control_point, contour, id))
        else:
            contours.insert(0, Contour(control_point, contour, id))

    vessel.add_contours(contours)

    return vessel
