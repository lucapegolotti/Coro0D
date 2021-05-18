from vessel_portion import VesselPortion
from models import *

class PhysicalBlock:
    def __init__(self, vessel_portion, model_type):
        self.portion = vessel_portion
        if model_type == "Windkessel2":
            self.model = Windkessel2(vessel_portion.compute_R(),
                                     vessel_portion.compute_C())
        else:
            raise NotImplementedError(model_type + " not implemented")


def create_physical_blocks(portions, model_type):
    physical_blocks = []
    for portion in portions:
        physical_blocks += [PhysicalBlock(portion, model_type)]

    return physical_blocks
