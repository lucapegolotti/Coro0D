import math
import numpy as np


class Contour:
    def __init__(self, control_point, contour, id_path):
        self.control_point = control_point
        self.contour = contour
        self.id_path = id_path

        npoints = contour.shape[0]
        center = contour[0, :]

        crossp = np.zeros([1, 3])
        # we calculate the area using this formula
        # https://math.stackexchange.com/questions/3207981/caculate-area-of-polygon-in-3d
        for icont in range(1, npoints - 1):
            crossp += np.cross(contour[icont, :] - center, contour[icont + 1, :] - center)

        crossp += np.cross(contour[-1, :] - center, contour[0, :] - center)
        crossp *= 0.5
        area = np.linalg.norm(crossp)

        # we compute the radius by assuming a circular contour
        self.area = area
        self.radius = np.sqrt(area / math.pi)

        return
