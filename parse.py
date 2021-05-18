import os
import re
import numpy as np
from numpy import linalg
import xml.etree.ElementTree as ET
from vessel_portion import VesselPortion
from contour import Contour
from constants import *

def parse_vessels(fdr):
    fullpaths = []
    for filename in os.listdir(fdr + "Paths/"):
        path = parse_single_path(fdr + "Paths/" + filename)
        filenamectgr = filename[0:-4] + ".ctgr"
        path = parse_single_segmentation(fdr + "Segmentations/" + filenamectgr, path)
        fullpaths.append(path)

    return fullpaths

def open_xml(namefile):
    with open(namefile) as f:
        xml = f.read()

    # Add fake root to avoid problems due to presence of multiple roots
    tree = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) +
                         "</root>")

    return tree

def parse_single_path(namefile):
    tree = open_xml(namefile)

    xs = []
    ys = []
    zs = []

    path_points = tree[1][0][0][1]
    for child in path_points:
        x = float(child[0].attrib['x'])
        y = float(child[0].attrib['y'])
        z = float(child[0].attrib['z'])
        xs.append(x)
        ys.append(y)
        zs.append(z)

    single_path = VesselPortion(xs, ys, zs)
    return single_path

def parse_single_segmentation(namefile, vessel):
    tree = open_xml(namefile)

    contours = []
    # we skip the first index because it corresponds to "lofting_parameters"
    ncontours = len(tree[1][0]) - 1
    for icont in range(0, ncontours):
        curcontour = tree[1][0][icont+1]
        x = float(curcontour[0][0].attrib['x'])
        y = float(curcontour[0][0].attrib['y'])
        z = float(curcontour[0][0].attrib['z'])
        id = int(curcontour[0].attrib['id'])
        control_point = np.array([x,y,z])
        curpoints = curcontour[2]
        ncurpoints = len(curpoints)
        contour = np.zeros([ncurpoints,3])
        for ipoint in range(0, ncurpoints):
            contour[ipoint,0] = curpoints[ipoint].attrib['x']
            contour[ipoint,1] = curpoints[ipoint].attrib['y']
            contour[ipoint,2] = curpoints[ipoint].attrib['z']

        contours.append(Contour(control_point, contour, id))

    vessel.add_contours(contours)

    return vessel
