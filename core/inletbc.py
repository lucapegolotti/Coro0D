import csv
import math
import numpy as np
import os
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter


class InletBC:
    def __init__(self, portion, index, bc_type, folder, problem_data):
        self.portion = portion
        self.index = index
        self.bc_type = bc_type
        self.file = os.path.join(folder, os.path.normpath("Data/measures.csv"))
        self.problem_data = problem_data
        self.parse_inlet_function()

        # self.pressure_values = np.zeros(0)
        # self.times = np.zeros(0)
        # self.pressurespline = tuple()
        # self.indices_minpressures = np.zeros(0, dtype=np.int)

        return

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

            # discarding local minima that are too close (pick only the smaller one)
            Tmin = 0.4
            Nmin = int(Tmin / samsize)
            for cnt in range(len(minima)):
                if np.sum(minima[max(0, cnt-Nmin):min(cnt+Nmin, len(minima)-1)]) > 1:
                    true_minima_index = np.argmin(filtered_pressures[max(0, cnt-Nmin):min(cnt+Nmin, len(minima)-1)]) +\
                                        max(0, cnt-Nmin)
                    minima[max(0, cnt-Nmin):min(cnt+Nmin, len(minima)-1)] = False
                    minima[true_minima_index] = True
            minima = np.where(minima)[0]

            # discarding minima that are too high (compared to the previous and the next)
            minima_values = filtered_pressures[minima]
            valid_minima = np.zeros_like(minima_values, dtype=bool)
            for cnt, value in enumerate(minima_values):
                threshold_value = 1.2*np.min(minima_values[max(0, cnt-1):min(cnt+1, len(minima_values)-1)])
                valid_minima[cnt] = (value <= threshold_value)
            minima = minima[valid_minima]

            # discarding minima occurring before t0
            minima = minima[minima >= self.problem_data.t0 / samsize]

            # this is to check that the minima are visually correct
            # import matplotlib.pyplot as plt
            # plt.figure()
            # ax = plt.axes()
            # ax.plot(self.times, self.pressure_values)
            # ax.plot(self.times[minima], self.pressure_values[minima], 'ro')
            # plt.show()

            if len(minima) <= self.problem_data.starting_minima:
                raise ValueError(f"Invalid starting minima index! "
                                 f"In the interval [{self.problem_data.t0}, {self.problem_data.T}] the pressure "
                                 f"exhibits only {len(minima)} minima, so it is not possible to start from  the minima "
                                 f"number {self.problem_data.starting_minima}!")

            self.pressure_values = self.pressure_values[minima[self.problem_data.starting_minima]:]
            self.times = np.subtract(self.times[minima[self.problem_data.starting_minima]:],
                                     self.times[minima[self.problem_data.starting_minima]])

            self.pressurespline = splrep(self.times, self.pressure_values)
            self.indices_minpressures = np.subtract(minima,
                                                    minima[self.problem_data.starting_minima])

        else:
            raise NotImplementedError("Inlet flowrate case not implemented")

        return

    def apply_bc_matrix_dot(self, matrix_dot, row):
        return

    def apply_bc_matrix(self, matrix, row):
        if self.bc_type == "pressure":
            matrix[row, self.index * 3 + 0] = -1
        elif self.bc_type == "flowrate":
            matrix[row, self.index * 3 + 2] = -1
        else:
            raise NotImplementedError(self.bc_type + " bc not implemented")

        return

    def apply_bc_vector(self, vector, time, row):
        vector[row] = self.inlet_function(time)
        return

    def evaluate_ramp(self, time):
        t0 = self.problem_data.t0
        t0ramp = self.problem_data.t0ramp
        target = splev(t0, self.pressurespline, der=0)
        return target * (1.0 - math.cos((time - t0ramp) * math.pi / (t0 - t0ramp))) / 2.0

    def inlet_function(self, time):
        if time < self.problem_data.t0:
            return self.evaluate_ramp(time)
        return splev(time, self.pressurespline, der=0)
