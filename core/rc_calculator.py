import numpy as np
import os

from connectivity import find_downstream_outlets_portions


class RCCalculator:
    def __init__(self, folder, coronary, coeff_resistance=1, coeff_capacitance=1):
        self.folder = folder
        self.coronary = coronary
        self.coeff_resistance = coeff_resistance
        self.coeff_capacitance = coeff_capacitance
        self.compute_total_resistance()
        if self.coronary == "left":
            self.total_capacitance = 3.6 * 1e-5 * self.coeff_capacitance
        elif self.coronary == "right":
            self.total_capacitance = 2.5 * 1e-5 * self.coeff_capacitance
        else:
            raise ValueError("Coronary type must be marked as 'left' or 'right'")

        return

    def compute_total_resistance(self):
        # cardiac output in ml/s
        co = open(os.path.join(self.folder, os.path.normpath("Data/cardiac_output.txt")), "r")
        Q = float(co.readline()) * 1000 / 60
        # average pressure in dyn/cm^2
        map = open(os.path.join(self.folder, os.path.normpath("Data/mean_aortic_pressure.txt")), "r")
        mean_Pa = float(map.readline()) * 1333.2

        systemic_resistance = mean_Pa / Q
        coronary_total_resistance = systemic_resistance * 25 * self.coeff_resistance

        gamma = 7 / 3

        if self.coronary == "left":
            self.total_resistance = (1 + gamma) / gamma * coronary_total_resistance
        elif self.coronary == "right":
            self.total_resistance = (1 + gamma) * coronary_total_resistance
        else:
            raise ValueError("Coronary type must be marked as 'left' or 'right'")

        return

    def assign_resistances_to_outlets(self, portions, connectivity):
        maxoutletflag = int(np.max(connectivity))

        m = 2.6  # murray's law of coronaries
        suma = 0
        for flag in range(3, maxoutletflag + 1):
            portionindex = int(np.where(connectivity == flag)[1])
            curarea = portions[portionindex].compute_area_outlet()
            suma += np.sqrt(curarea) ** m

        for flag in range(3, maxoutletflag + 1):
            portionindex = int(np.where(connectivity == flag)[1])
            curarea = portions[portionindex].compute_area_outlet()
            curresistance = suma / (np.sqrt(curarea) ** m) * self.total_resistance
            portions[portionindex].set_total_outlet_resistance(curresistance)

        return

    def assign_capacitances_to_outlets(self, portions, connectivity):
        maxoutletflag = int(np.max(connectivity))

        suma = 0
        for flag in range(3, maxoutletflag + 1):
            portionindex = int(np.where(connectivity == flag)[1])
            curarea = portions[portionindex].compute_area_outlet()
            suma += curarea

        for flag in range(3, maxoutletflag + 1):
            portionindex = int(np.where(connectivity == flag)[1])
            curarea = portions[portionindex].compute_area_outlet()
            curresistance = curarea / suma * self.total_capacitance
            portions[portionindex].set_total_outlet_capacitance(curresistance)

        return

    def assign_downstream_resistances(self, portions, connectivity):

        outlet_portion_indices = np.where(connectivity > 2)[1]
        if any([portions[idx].total_outlet_resistance is None for idx in outlet_portion_indices]):
            self.assign_resistances_to_outlets(portions, connectivity)

        for portionindex in range(len(portions)):
            outlet_portions = []
            find_downstream_outlets_portions(portions, portionindex, connectivity, outlet_portions)

            R = 0.0
            for outlet_portion in outlet_portions:
                R += 1.0 / outlet_portion.total_outlet_resistance

            R = 1.0 / R

            portions[portionindex].set_downstream_outlet_resistance(R)

        return
