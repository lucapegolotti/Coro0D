import os

from models import *
from connectivity import find_neighbours
from inletbc import InletBC


class PhysicalBlock:
    def __init__(self, vessel_portion, model_type, problem_data, other_data=None):

        self.portion = vessel_portion
        self.isStenotic = self.portion.isStenotic

        # NON-STENOTIC MODELS
        if model_type == "R_model":
            self.model = R_model(vessel_portion, problem_data)
        elif model_type == "RC_model":
            self.model = RC_model(vessel_portion, problem_data)
        elif model_type == "RL_model":
            self.model = RL_model(vessel_portion, problem_data)
        elif model_type == "RLC_model":
            self.model = RLC_model(vessel_portion, problem_data)
        elif model_type == "Windkessel2":
            self.model = Windkessel2(vessel_portion, problem_data)

        # STENOTIC MODELS
        elif model_type == "YoungTsai":
            assert 'r0' in other_data.keys()
            self.model = YoungTsai(vessel_portion, problem_data, other_data['r0'])
        elif model_type == "Garcia":
            assert 'r0' in other_data.keys()
            self.model = Garcia(vessel_portion, problem_data, other_data['r0'])
        elif model_type == "ItuSharma":
            assert 'r0' in other_data.keys()
            assert 'HR' in other_data.keys()
            # assert 'CO' in other_data.keys()
            # assert 'MAP' in other_data.keys()
            self.model = ItuSharma(vessel_portion, problem_data,
                                   other_data['r0'], other_data['HR'],
                                   other_data['sol_steady'])
        elif model_type == "ResistanceStenosis":
            assert 'r0' in other_data.keys()
            self.model = ResistanceStenosis(vessel_portion, problem_data, other_data['r0'])
        else:
            raise NotImplementedError(model_type + " not implemented!")

        return


def create_physical_blocks(portions, model_type, stenosis_model_type, problem_data,
                           folder=None, connectivity=None, sol_steady=None):
    physical_blocks = []
    for (index_portion, portion) in enumerate(portions):
        if not portion.isStenotic:
            physical_blocks += [PhysicalBlock(portion, model_type, problem_data)]
        else:

            other_data = dict()

            if stenosis_model_type == "None":
                portion.isStenotic = False
            else:

                assert connectivity is not None
                neighs = find_neighbours(portions, index_portion, problem_data.tol)
                r0 = 0
                if len(neighs['IN']) == 0:
                    raise ValueError("Invalid to have stenosis at the inlet portion!")
                for in_neigh in neighs['IN']:
                    portions[in_neigh].compute_mean_radius()
                    if portions[in_neigh].mean_radius > r0:
                        r0 = portions[in_neigh].mean_radius
                other_data['r0'] = r0

                assert folder is not None
                HR = InletBC.compute_HR(folder, problem_data)
                other_data['HR'] = HR

                if sol_steady is not None:
                    other_data['sol_steady'] = sol_steady[index_portion*4:(index_portion+1)*4]
                else:
                    other_data['sol_steady'] = None

                # co = open(os.path.join(folder, os.path.normpath("Data/cardiac_output.txt")), "r")
                # CO = float(co.readline()) * (1000 / 60)  # conversion from L/min to mL/s
                # CO *= 0.04  # 4% of flow goes in coronaries
                # CO *= (0.7 if problem_data.side == "left" else 0.3 if problem_data.side == "right" else 0.0)
                # other_data['CO'] = CO
                #
                # file = open(os.path.join(folder, os.path.normpath("Data/mean_aortic_pressure.txt")), "r")
                # MAP = float(file.readline()) * 1333.2  # conversion from mmHg to dyn/cm^2
                # other_data['MAP'] = MAP

            physical_blocks += [PhysicalBlock(portion,
                                              stenosis_model_type if portion.isStenotic else model_type,
                                              problem_data, other_data)]

    return physical_blocks
