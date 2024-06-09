from enum import Enum
from typing import NamedTuple, Union

class InteractionType(Enum):
    NN = "nn"
    NNN = "nnn"
    ALL2ALL = "all2all"

class Criticality(Enum):
    CLOSE = "close"
    FAR = "far"


class InteractionsInput(NamedTuple):
    interaction_type: InteractionType = InteractionType.ALL2ALL
    Omega: Union[int, float] = 1.0
    rydberg_blockade: Union[int, float] = pow(7, 1/6)


class InvalidInteractionError(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        interaction_type -- input interaction_type which caused the error
        error_message -- explanation of the error
    """

    def __init__(self, interaction_type, error_message="Invalid interaction type"):
        self.interaction_type = interaction_type
        self.err = error_message
        super().__init__(self.err)