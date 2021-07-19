from models import *
from connectivity import find_neighbours
from inletbc import InletBC


class PhysicalBlock:
    def __init__(self, vessel_portion, model_type, problem_data, other_data=None):

        self.portion = vessel_portion
        self.isStenotic = self.portion.isStenotic

        # NON-STENOSIS MODEL
        if model_type == "Windkessel2":
            # self.model = Windkessel2(vessel_portion.compute_R(problem_data.viscosity),
            #                          vessel_portion.compute_C(problem_data.E,
            #                                                   problem_data.thickness_ratio))
            self.model = Windkessel2(vessel_portion, problem_data)
        elif model_type == "Resistance":
            # self.model = Resistance(vessel_portion.compute_R(problem_data.viscosity))
            self.model = Resistance(vessel_portion, problem_data)

        # STENOSIS MODELS
        elif model_type == "YoungTsai":
            assert 'r0' in other_data.keys()
            # self.model = YoungTsai(vessel_portion.compute_R_YT(problem_data.viscosity,
            #                                                    other_data['r0']),
            #                        vessel_portion.compute_R2_YT(problem_data.density,
            #                                                     other_data['r0']),
            #                        vessel_portion.compute_L_YT(problem_data.density,
            #                                                    other_data['r0']))
            self.model = YoungTsai(vessel_portion, problem_data, other_data['r0'])
        elif model_type == "Garcia":
            assert 'r0' in other_data.keys()
            # self.model = Garcia(vessel_portion.compute_R2_G(problem_data.density,
            #                                                 other_data['r0']),
            #                     vessel_portion.compute_L_G(problem_data.density,
            #                                                other_data['r0']))
            self.model = Garcia(vessel_portion, problem_data, other_data['r0'])
        elif model_type == "ItuSharma":
            assert 'r0' in other_data.keys()
            assert 'HR' in other_data.keys()
            # self.model = ItuSharma(vessel_portion.compute_R_IS(problem_data.density,
            #                                                    problem_data.viscosity,
            #                                                    other_data['HR'],
            #                                                    other_data['r0']),
            #                        vessel_portion.compute_R2_IS(problem_data.density,
            #                                                     other_data['r0']),
            #                        vessel_portion.compute_L_IS(problem_data.density))
            self.model = ItuSharma(vessel_portion, problem_data, other_data['r0'], other_data['HR'])
        elif model_type == "ResistanceStenosis":
            assert 'r0' in other_data.keys()
            # self.model = ResistanceStenosis(vessel_portion.compute_R(problem_data.viscosity),
            #                                 vessel_portion.compute_R2(problem_data.density,
            #                                                           other_data['r0']))
            self.model = ResistanceStenosis(vessel_portion, problem_data, other_data['r0'])
        elif model_type == "Windkessel2Stenosis":
            assert 'r0' in other_data.keys()
            # self.model = Windkessel2Stenosis(vessel_portion.compute_R(problem_data.viscosity),
            #                                  vessel_portion.compute_C(problem_data.E,
            #                                                           problem_data.thickness_ratio),
            #                                  vessel_portion.compute_R2(problem_data.density,
            #                                                            other_data['r0']))
            self.model = Windkessel2Stenosis(vessel_portion, problem_data, other_data['r0'])
        else:
            raise NotImplementedError(model_type + " not implemented!")

        return


def create_physical_blocks(portions, model_type, stenosis_model_type, problem_data,
                           folder=None, connectivity=None):
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

            physical_blocks += [PhysicalBlock(portion,
                                              stenosis_model_type if portion.isStenotic else model_type,
                                              problem_data, other_data)]

    return physical_blocks
