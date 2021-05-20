import numpy as np

class ResistanceCalculator:
    def __init__(self, folder, coronary):
        self.folder = folder
        self.coronary = coronary
        self.compute_total_resistance()

    def compute_total_resistance(self):
        co = open(self.folder + "/Data/cardiac_output.txt", "r")
        Q = float(co.readline())

        map = open(self.folder + "/Data/mean_aortic_pressure.txt", "r")
        mean_Pa = float(map.readline()) * 1333.2

        systemic_resistance = mean_Pa / Q
        coronary_total_resistance = systemic_resistance * 25

        gamma = 7/3

        if self.coronary == "left":
            self.total_resistance = (1 + gamma) / gamma * coronary_total_resistance
        elif self.coronary == "right":
            self.total_resistance = (1 + gamma) * coronary_total_resistance
        else:
            raise ValueError("coronary type must be left or right")

    def assign_resistances_to_outlets(self, portions, connectivity):
        maxoutletflag = int(np.max(connectivity))

        m = 2.6 # murray's law of coronaries
        suma = 0
        for flag in range(3, maxoutletflag + 1):
            portionindex = int(np.where(connectivity == flag)[1])
            curarea = portions[portionindex].compute_area_outlet()
            print(suma)
            print(curarea)
            suma += np.sqrt(curarea) ** m

        for flag in range(3, maxoutletflag + 1):
            portionindex = int(np.where(connectivity == flag)[1])
            curarea = portions[portionindex].compute_area_outlet()
            curresistance = suma / (np.sqrt(curarea) ** m) * self.total_resistance
            portions[portionindex].set_total_outlet_resistance(curresistance)
