import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep


class DistalPressureGenerator:
    def __init__(self, times, indexminima, folder, problem_data, coronary, coeff, shift):
        self.times = times
        self.indexminima = indexminima
        self.file = folder + "/Data/plv.dat"
        self.problem_data = problem_data
        self.shift = shift
        if coronary == "left":
            self.coeff = 1.5 * coeff
        if coronary == "right":
            self.coeff = 0.5 * coeff
        self.parse_myocardial_pressure()
        self.build_myocardial_pressure()

        return

    def parse_myocardial_pressure(self):
        self.myopressure_original = []
        self.times_original = []
        with open(self.file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t')
            for row in spamreader:
                try:
                    self.times_original.append(float(row[0]))
                    self.myopressure_original.append(float(row[1]) * self.coeff)
                except Exception:
                    pass

        self.times_original = np.array(self.times_original)
        self.myopressure_original = np.array(self.myopressure_original)

        # plot original and shifted myocardial pressure
        plt.figure()
        ax = plt.axes()
        original, = ax.plot(self.times_original, self.myopressure_original)
        # apply shift
        self.myopressure_original = np.roll(self.myopressure_original, self.shift)
        ax.set_xlim(0, 1)
        shifted, = ax.plot(self.times_original, self.myopressure_original,
                           color='red', linestyle='dashed')
        ax.set_title("Myocardial pressure")
        ax.legend([original, shifted], ['original', 'shifted'])

        self.myopressurespline_original = splrep(self.times_original, self.myopressure_original)

        return

    def build_myocardial_pressure(self):
        mm = self.indexminima
        self.myopressure = np.zeros(self.times.shape)
        nperiods = mm.shape[0] - 1
        original_period = self.times_original[-1] - self.times_original[0]

        for iperiod in range(nperiods):
            # period = self.times[mm[iperiod + 1]] - self.times[mm[iperiod]]
            for index in range(mm[iperiod], mm[iperiod + 1]):
                # we scale the current time to be in the original period
                scaledtime = self.times_original[0] + (self.times[index] - self.times[mm[iperiod]]) / \
                             (self.times[mm[iperiod + 1]] - self.times[mm[iperiod]]) * original_period
                self.myopressure[index] = splev(scaledtime, self.myopressurespline_original, der=0)

        # check how the myocardial pressure we built looks like
        # plt.figure()
        # ax = plt.axes()
        # ax.plot(self.times, self.myopressure)
        # ax.set_xlim(0,2)
        # plt.show()

        self.myopressurespline = splrep(self.times, self.myopressure)

        return

    def evaluate_ramp(self, time):
        t0ramp = self.problem_data.t0ramp
        t0 = self.problem_data.t0
        target = splev(t0, self.myopressurespline, der=0)
        return target * (1.0 - math.cos((time - t0ramp) * math.pi / (t0 - t0ramp))) / 2.0

    def distal_pressure(self, time):
        if time < self.problem_data.t0:
            return self.evaluate_ramp(time)
        return splev(time, self.myopressurespline, der=0)
