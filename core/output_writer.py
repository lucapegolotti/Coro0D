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
            fstr += str(float(self.portions[index].compute_Ramicro()) * coeff_r) + "\n"

            fstr += "REAL(8) :: Cim_" + self.portions[index].pathname + " = "
            fstr += str(float(self.portions[index].compute_Cim()) * coeff_c) + "\n"

            fstr += "REAL(8) :: Rv_" + self.portions[index].pathname + " = "
            fstr += str(float(self.portions[index].compute_Rv() +
                              self.portions[index].compute_Rvmicro()) * coeff_r) + "\n"

            fstr += "\n"

        outfile = open(os.path.join(self.output_fdr, "resistance_capacitance.txt"), "w")
        outfile.write(fstr)
        outfile.close()

        return

    def write_distal_pressure(self, npoints):
        if self.problem_data.units == "mm":
            coeff = 10
        else:
            coeff = 1

        # we create a fine partition of the period
        times = np.linspace(self.bc_manager.inletbc.t0_eff - self.problem_data.Tramp,
                            self.bc_manager.inletbc.T_eff, npoints)

        outfile = open(os.path.join(self.output_fdr, "distal_pressure.txt"), "w")
        for t in times:
            dp = self.bc_manager.distal_pressure_generator.distal_pressure(t)
            curstr = str(float(t - times[0])) + " "
            curstr += str(float(dp) * coeff) + "\n"
            outfile.write(curstr)
        outfile.close()

        return

    def write_inlet_pressure(self, npoints):
        if self.problem_data.units == "mm":
            coeff = 10
        else:
            coeff = 1

        # we create a fine partition of the period
        times = np.linspace(self.bc_manager.inletbc.t0_eff - self.problem_data.Tramp,
                            self.bc_manager.inletbc.T_eff, npoints)

        outfile = open(os.path.join(self.output_fdr, "inlet_pressure.txt"), "w")
        for t in times:
            p = self.bc_manager.inletbc.inlet_function(t)
            curstr = str(float(t - times[0])) + " "
            curstr += str(float(p) * coeff) + "\n"
            outfile.write(curstr)
        outfile.close()

        return

    def write_inlet_outlets_flow_pressures(self, times, solutions):
        labels = "time,"
        indices = [self.bc_manager.inletindex]
        indices += self.bc_manager.outletindices
        labels += self.portions[indices[0]].pathname + "_in,"

        for i in range(1, len(indices)):
            labels += self.portions[indices[i]].pathname + "_out,"

        labels = labels[:-1]

        flows = solutions[4 * np.array(indices) + 3, :]
        M = np.vstack((times, flows))
        np.savetxt(self.output_fdr + "/flows_res.csv",
                   M.T, delimiter=",",
                   header=labels)

        pressures1 = solutions[4 * np.array(self.bc_manager.inletindex) + 0, :]
        pressures2 = solutions[4 * np.array(self.bc_manager.outletindices) + 1, :]
        M = np.vstack((times, pressures1, pressures2))
        np.savetxt(self.output_fdr + "/pressure_res.csv",
                   M.T, delimiter=",",
                   header=labels)

        return

    def write_thickess_caps(self):
        outfile = open(self.output_fdr + "/thickness.txt", "w")
        for portion in self.portions:
            posindices = np.where(portion.radii > 0)
            posradii = portion.radii[posindices]
            curstr = portion.pathname + " "
            # we consider 10% of the radius for the membrane thickness
            curstr += str(2 * posradii[0] * 0.1) + " "
            curstr += str(2 * posradii[-1] * 0.1) + "\n"
            outfile.write(curstr)
        outfile.close()

        return

    def write_ffr_pressures(self, times, solutions, stenotic_portions, ffr_portions, dt=0.01):
        factor_dt = dt / self.problem_data.deltat
        if factor_dt - int(factor_dt) > 1e-4:
            raise ValueError("The target timestep dt must be an integer multiple of the simulation timestep!")
        factor_dt = int(factor_dt)
        index0 = np.where(np.abs(times - self.bc_manager.inletbc.t0_eff) < self.problem_data.deltat/2)[0][0]
        L = int((times[-1] - times[index0]) / dt) + 1
        out_times = times[index0::factor_dt]

        labels = 'time,'

        pressureIn = solutions[4 * self.bc_manager.inletindex + 0, index0::factor_dt]
        labels += 'Inlet,'

        pressuresOut = np.zeros((len(ffr_portions), L))
        cntPath = dict()
        cnt = 0
        for (stenotic_portion, ffr_portion) in zip(stenotic_portions, ffr_portions):
            pressuresOut[cnt, :] = solutions[4 * ffr_portion + 1, index0::factor_dt]
            pathname = self.portions[stenotic_portion].pathname
            if pathname in cntPath:
                cntPath[pathname] += 1
            else:
                cntPath[pathname] = 1
            labels += pathname + "-" + str(cntPath[pathname]) + ","
            cnt += 1

        labels = labels[:-1]

        retMat = np.vstack((out_times, pressureIn, pressuresOut))
        np.savetxt(self.output_fdr + "/pressure_ffr_0D.csv",
                   retMat.T, delimiter=",",
                   header=labels)


