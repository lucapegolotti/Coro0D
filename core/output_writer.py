import os
import numpy as np


class OutputWriter:
    def __init__(self, output_fdr, bc_manager, portions, problem_data):
        self.output_fdr = output_fdr
        if not os.path.isdir(output_fdr):
            try:
                os.mkdir(output_fdr)
            except OSError:
                print("Creation of the directory %s failed" % output_fdr)

        self.problem_data = problem_data
        self.bc_manager = bc_manager
        self.portions = portions

        return

    def write_outlet_rc(self):
        outlet_indices = self.bc_manager.outletindices

        # the resistance is now in cgs but we want to convert it to mgs because
        # that is likely the system we are using in SimVascular
        if self.problem_data.units == "mm":
            coeff_r = 10 ** 4
        else:
            coeff_r = 1

        if self.problem_data.units == "mm":
            coeff_c = 10 ** (-4)
        else:
            coeff_c = 1

        fstr = ""

        for index in outlet_indices:
            fstr += "REAL(8) :: Ra_" + self.portions[index].pathname + " = "
            fstr += str(float(self.portions[index].compute_Ra()) * coeff_r) + "\n"

            fstr += "REAL(8) :: Ca_" + self.portions[index].pathname + " = "
            fstr += str(float(self.portions[index].compute_Ca()) * coeff_c) + "\n"

            fstr += "REAL(8) :: Rmicro_" + self.portions[index].pathname + " = "
            fstr += str(float(self.portions[index].compute_Ramicro() + \
                              self.portions[index].compute_Rvmicro()) * coeff_r) + "\n"

            fstr += "REAL(8) :: Cim_" + self.portions[index].pathname + " = "
            fstr += str(float(self.portions[index].compute_Cim()) * coeff_c) + "\n"

            fstr += "REAL(8) :: Rv_" + self.portions[index].pathname + " = "
            fstr += str(float(self.portions[index].compute_Rv()) * coeff_r) + "\n"

            fstr += "\n"

        outfile = open(os.path.join(self.output_fdr, "resistance_capacitance.txt"), "w")
        outfile.write(fstr)
        outfile.close()

        return

    def write_distal_pressure(self, data, npoints):
        if self.problem_data.units == "mm":
            coeff = 10
        else:
            coeff = 1

        # we create a fine partition of the period
        times = np.linspace(data.t0ramp, data.T, npoints)

        outfile = open(os.path.join(self.output_fdr, "distal_pressure.txt"), "w")
        for t in times:
            dp = self.bc_manager.distal_pressure_generator.distal_pressure(t)
            curstr = str(float(t - times[0])) + " "
            curstr += str(float(dp) * coeff) + "\n"
            outfile.write(curstr)
        outfile.close()

        return

    def write_inlet_pressure(self, data, npoints):
        if self.problem_data.units == "mm":
            coeff = 10
        else:
            coeff = 1

        # we create a fine partition of the period
        times = np.linspace(data.t0ramp, data.T, npoints)

        outfile = open(os.path.join(self.output_fdr, "inlet_pressure.txt"), "w")
        for t in times:
            p = self.bc_manager.inletbc.inlet_function(t)
            curstr = str(float(t - times[0])) + " "
            curstr += str(float(p) * coeff) + "\n"
            outfile.write(curstr)
        outfile.close()

        return
        
    def write_inlet_outlets_flow_pressures(self, times, solutions, portions, bc_manager):
        labels = "time,"
        indices = [bc_manager.inletindex]
        indices += bc_manager.outletindices
        labels += portions[indices[0]].pathname + "_in,"

        for i in range(1, len(indices)):
            labels += portions[indices[i]].pathname + "_out,"

        labels = labels[:-1]

        flows = solutions[3 * np.array(indices) + 2, :]
        M = np.vstack((times,flows))
        np.savetxt(self.output_fdr + "/flows_res.csv",
                   M.T, delimiter=",",
                   header = labels)

        pressures1 = solutions[3 * np.array(bc_manager.inletindex) + 0, :]
        pressures2 = solutions[3 * np.array(bc_manager.outletindices) + 1, :]
        M = np.vstack((times,pressures1,pressures2))
        np.savetxt(self.output_fdr + "/pressure_res.csv",
                   M.T, delimiter=",",
                   header = labels)

    def write_thickess_caps(self, portions):
        outfile = open(self.output_fdr + "/thickness.txt", "w")
        for portion in portions:
            posindices = np.where(portion.radii > 0)
            posradii = portion.radii[posindices]
            curstr = portion.pathname + " "
            # we consider 10% of the radius for the membrane thickness
            curstr += str(2 * posradii[0] * 0.1) + " "
            curstr += str(2 * posradii[-1] * 0.1) + "\n"
            outfile.write(curstr)
        outfile.close()
