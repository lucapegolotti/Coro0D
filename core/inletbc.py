import csv
import math
import numpy as np
import os
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter
from scipy.integrate import simps


class InletBC:
    def __init__(self, portion, index, bc_type, folder, problem_data):
        self.portion = portion
        self.index = index
        self.bc_type = bc_type
        self.file = os.path.join(folder, os.path.normpath("Data/measures.csv"))
        # self.MAP_file = os.path.join(folder, os.path.normpath("Data/mean_aortic_pressure.txt"))
        self.problem_data = problem_data
        self.parse_inlet_function()

        return

    def parse_inlet_function(self):
        if self.problem_data.use_inlet_pressure:
            samsize = 0.01
            self.pressure_values = []
            self.times = [-samsize]
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

            # we find the local minima. First we apply a lowpass filter
            filtered_pressures = savgol_filter(self.pressure_values, 13, 2)
            minima = np.r_[True, filtered_pressures[1:] < filtered_pressures[:-1]] & \
                     np.r_[filtered_pressures[:-1] < filtered_pressures[1:], True]

            # discarding local minima that are too close (pick only the smaller one)
            Tmin = 0.3
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
                threshold_value = 1.25*np.min(minima_values[max(0, cnt-1):min(cnt+1, len(minima_values)-1)])
                valid_minima[cnt] = (value <= threshold_value)
            minima = minima[valid_minima]

            # discarding minima occurring before t0
            minima = minima[minima >= self.problem_data.t0 / samsize]

            # this is to check that the minima are visually correct
            # import matplotlib.pyplot as plt
            # plt.figure()
            # ax = plt.axes()
            # ax.plot(self.times, filtered_pressures / 1333.22)
            # ax.plot(self.times[minima], self.pressure_values[minima] / 1333.22, 'ro')
            # ax.set_ylim([40, 160])
            # plt.show()

            if len(minima) <= self.problem_data.starting_minima:
                raise ValueError(f"Invalid starting minima index! "
                                 f"In the interval [{self.problem_data.t0}, {self.problem_data.T}] the pressure "
                                 f"exhibits only {len(minima)} minima, so it is not possible to start from the minima "
                                 f"number {self.problem_data.starting_minima}!")

            self.t0_eff = self.times[minima[self.problem_data.starting_minima]]
            self.T_eff = self.t0_eff + self.problem_data.T
            shift = 0
            while self.times[minima[self.problem_data.starting_minima + shift]] <= self.T_eff:
                shift += 1
            self.n_heartbeats = shift - 1
            self.T_hb = self.times[minima[self.problem_data.starting_minima + self.n_heartbeats]]
            self.HR = self.n_heartbeats / (self.T_hb - self.t0_eff) * 60.0
            print(f"\n  TIMES SUMMARY  ")
            print(f"Effective initial time: {self.t0_eff} s")
            print(f"Effective final time: {self.T_eff} s")
            print(f"Number of complete heartbeats: {self.n_heartbeats}")
            print(f"Heart rate: {self.HR:.2f} bpm\n")

            self.pressure_values = self.pressure_values[minima[self.problem_data.starting_minima]:]
            self.times = self.times[minima[self.problem_data.starting_minima]:]
            # self.times = np.subtract(self.times[minima[self.problem_data.starting_minima]:],
            #                          self.times[minima[self.problem_data.starting_minima]])

            self.pressurespline = splrep(self.times, self.pressure_values)
            self.indices_minpressures = np.subtract(minima,
                                                    minima[self.problem_data.starting_minima])

            # file = open(self.MAP_file, "r")
            # self.MAP = float(file.readline()) * 1333.2
            indices = [0, int(self.problem_data.T / samsize)]
            self.MAP = simps(self.pressure_values[indices[0]:indices[1]], self.times[indices[0]:indices[1]]) / \
                       (self.times[indices[1]] - self.times[indices[0]])

        else:
            raise NotImplementedError("Inlet flowrate case not implemented")

        return

    def apply_bc_matrix_dot(self, matrix_dot, row):
        return

    def apply_bc_matrix(self, matrix, row):
        if self.bc_type == "pressure":
            matrix[row, self.index * 4 + 0] = -1
        elif self.bc_type == "flowrate":
            matrix[row, self.index * 4 + 2] = -1
        else:
            raise NotImplementedError(self.bc_type + " bc not implemented")

        return

    def apply_bc_vector(self, vector, time, row, steady=False):
        if not steady:
            vector[row] = self.inlet_function(time)
        else:
            vector[row] = self.MAP
        return

    def apply_0bc_vector(self, vector, time, row):
        vector[row] = 0
        return

    def evaluate_ramp(self, time):
        Tramp = self.problem_data.Tramp
        t0ramp = self.t0_eff - Tramp
        t0 = self.t0_eff
        target = splev(t0, self.pressurespline, der=0)
        return target * (1.0 - math.cos((time - t0ramp) / (t0 - t0ramp) * math.pi)) / 2.0

    def inlet_function(self, time):
        if time < self.t0_eff:
            return self.evaluate_ramp(time)
        return splev(time, self.pressurespline, der=0)

    @staticmethod
    def compute_HR(folder, problem_data):
        file = os.path.join(folder, os.path.normpath("Data/measures.csv"))
        samsize = 0.01
        pressure_values = []
        with open(file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                # we do this to avoid converting the header of the csv
                try:
                    # we multiply by 1333.2 because the pressure is in mmHg
                    # but we use cgs
                    pressure_values.append(float(row[0].split('\t')[0]) * 1333.2)
                except Exception:
                    pass
        pressure_values = np.array(pressure_values)

        # we find the local minima. First we apply a lowpass filter
        filtered_pressures = savgol_filter(pressure_values, 13, 2)
        minima = np.r_[True, filtered_pressures[1:] < filtered_pressures[:-1]] & \
                 np.r_[filtered_pressures[:-1] < filtered_pressures[1:], True]

        # discarding local minima that are too close (pick only the smaller one)
        Tmin = 0.3
        Nmin = int(Tmin / samsize)
        for cnt in range(len(minima)):
            if np.sum(minima[max(0, cnt - Nmin):min(cnt + Nmin, len(minima) - 1)]) > 1:
                true_minima_index = np.argmin(filtered_pressures[max(0, cnt-Nmin):min(cnt+Nmin, len(minima) - 1)]) + \
                                    max(0, cnt-Nmin)
                minima[max(0, cnt-Nmin):min(cnt+Nmin, len(minima) - 1)] = False
                minima[true_minima_index] = True
        minima = np.where(minima)[0]

        # discarding minima that are too high (compared to the previous and the next)
        minima_values = filtered_pressures[minima]
        valid_minima = np.zeros_like(minima_values, dtype=bool)
        for cnt, value in enumerate(minima_values):
            threshold_value = 1.25 * np.min(minima_values[max(0, cnt - 1):min(cnt + 1, len(minima_values) - 1)])
            valid_minima[cnt] = (value <= threshold_value)
        minima = minima[valid_minima]

        # discarding minima occurring before t0 and after T
        minima = minima[(minima >= problem_data.t0 / samsize) &
                        (minima <= (problem_data.t0 + problem_data.T) / samsize)]
        HR = len(minima) / ((minima[-1] - minima[0]) * samsize) * 60.0

        return HR
