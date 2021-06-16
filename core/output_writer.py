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
