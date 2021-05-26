from vessel_portion import VesselPortion
from models import *

class PhysicalBlock:
    def __init__(self, vessel_portion, model_type, problem_data):
        self.portion = vessel_portion
        if model_type == "Windkessel2":
            self.model = Windkessel2(vessel_portion.compute_R(problem_data.viscosity),
                                     vessel_portion.compute_C(problem_data.E, \
                                                              problem_data.thickness_ratio))
        elif model_type == "resistance":
            self.model = Resistance(vessel_portion.compute_R(problem_data.viscosity))
        else:
            raise NotImplementedError(model_type + " not implemented")


def create_physical_blocks(portions, model_type, problem_data):
    physical_blocks = []
    for portion in portions:
        physical_blocks += [PhysicalBlock(portion, model_type, problem_data)]

    return physical_blocks
