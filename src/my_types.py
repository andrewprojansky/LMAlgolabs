import enum
from typing import Generic, TypeVar, Union

T = TypeVar('T')  # Type of the value in case of success
E = TypeVar('E')  # Type of the error in case of failure

class Result(Generic[T, E]):
    def __init__(self, is_ok: bool, value: Union[T, E]):
        self.is_ok = is_ok
        self.value = value

    def __repr__(self):
        if self.is_ok:
            return f"Ok({self.value})"
        else:
            return f"Err({self.value})"

    def is_ok(self) -> bool:
        return self.is_ok

    def is_err(self) -> bool:
        return not self.is_ok

    def unwrap(self) -> T:
        if self.is_ok:
            return self.value
        else:
            raise ValueError(f"Called `unwrap` on an `Err` value: {self.value}")

    def unwrap_err(self) -> E:
        if not self.is_ok:
            return self.value
        else:
            raise ValueError(f"Called `unwrap_err` on an `Ok` value: {self.value}")

class Ok(Result):
    def __init__(self, value: T):
        super().__init__(True, value)

class Err(Result):
    def __init__(self, value: E):
        super().__init__(False, value)


class SaveType(enum.Enum):
    JSON = "json"
    PICKLE = "pkl"


class InteractionType(enum.Enum):
    NN = "nn"
    NNN = "nnn"
    ALL2ALL = "all2all"


