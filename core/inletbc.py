from vessel_portion import VesselPortion
import csv
import math
import numpy as np
from scipy import interpolate
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter

class InletBC:
    def __init__(self, portion, index, bc_type, folder, problem_data):
        self.portion = portion
        self.index = index
        self.bc_type = bc_type
        self.file = folder + "/Data/measures.csv"
        self.problem_data = problem_data
        self.parse_inlet_function()

    def parse_inlet_function(self):
        if self.problem_data.use_inlet_pressure:
            samsize = 0.01
            self.pressure_values = []
            self.times = [self.problem_data.t0 - samsize]
            with open(self.file, newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in spamreader:
                    # we do this to avoid converting the header of the csv
                    try:
                        # we multiply by 1333.2 because the pressure is in mmHg
                        # but we use cgs
                        self.pressure_values.append(float(row[0].split('\t')[0]) * 1333.2)
                        self.times.append(self.times[-1] + samsize)
                    except Exception:
                        pass

            self.times.pop(0)

            self.pressure_values = np.array(self.pressure_values)
            self.times = np.array(self.times)

            # we find the local minima. First we apply a filter

            filtered_pressures = savgol_filter(self.pressure_values, 13, 2)
            minima = np.r_[True, filtered_pressures[1:] < filtered_pressures[:-1]] & \
                     np.r_[filtered_pressures[:-1] < filtered_pressures[1:], True]
            minima = np.where(minima == True)[0]

            # this is to check that the minima are visually correct
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = plt.axes()
            # ax.plot(self.times, self.pressure_values)
            # ax.plot(self.times[minima], self.pressure_values[minima],'ro')
            # plt.xlim([0, 3])
            # plt.show()

            self.pressure_values = self.pressure_values[minima[0]:]
            self.times = np.subtract(self.times[minima[0]:],self.times[minima[0]])

            self.pressurespline = interpolate.splrep(np.array(self.times),
                                                     np.array(self.pressure_values))
            self.indices_minpressures = np.subtract(minima, minima[0])
        else:
            raise NotImplementedError("Inlet flowrate case not implemented")

    def apply_bc_matrix_dot(self, matrix_dot, row):
        return

    def apply_bc_matrix(self, matrix, row):
        if self.bc_type == "pressure":
            matrix[row,self.index * 3 + 0] = -1
        elif self.bc_type == "flowrate":
            matrix[row,self.index * 3 + 2] = -1
        else:
            raise NotImplementedError(self.bc_type + " bc not implemented")

    def apply_bc_vector(self, vector, time, row):
        vector[row] = self.inlet_function(time)

    def evaluate_ramp(self, time):
        t0ramp = self.problem_data.t0ramp
        t0 = self.problem_data.t0
        target = interpolate.splev(t0, self.pressurespline, der=0)
        return target * (1.0 - math.cos((time - t0ramp) * math.pi / (t0 - t0ramp))) / 2.0

    def inlet_function(self, time):
        if time < self.problem_data.t0:
            return self.evaluate_ramp(time)
        return interpolate.splev(time, self.pressurespline, der=0)
