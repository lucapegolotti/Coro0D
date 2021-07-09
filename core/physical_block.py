from models import *
from connectivity import find_neighbours


class PhysicalBlock:
    def __init__(self, vessel_portion, model_type, problem_data, other_data=None):

        self.portion = vessel_portion
        if model_type == "Windkessel2":
            self.model = Windkessel2(vessel_portion.compute_R(problem_data.viscosity),
                                     vessel_portion.compute_C(problem_data.E,
                                                              problem_data.thickness_ratio))
        elif model_type == "Resistance":
            self.model = Resistance(vessel_portion.compute_R(problem_data.viscosity))
        elif model_type == "Stenosis":
            assert 'r0' in other_data.keys()
            self.model = Stenosis(vessel_portion.compute_R(problem_data.viscosity),
                                  vessel_portion.compute_R2(problem_data.density,
                                                            other_data['r0']))
        else:
            raise NotImplementedError(model_type + " not implemented!")

        return


def create_physical_blocks(portions, model_type, problem_data, connectivity=None):
    physical_blocks = []
    for (index_portion, portion) in enumerate(portions):
        if not portion.isStenotic:
            physical_blocks += [PhysicalBlock(portion, model_type, problem_data)]
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
            other_data = {'r0': r0}

            physical_blocks += [PhysicalBlock(portion, "Stenosis", problem_data, other_data)]

    return physical_blocks
