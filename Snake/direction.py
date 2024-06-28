from enum import Enum, auto


class Direction(Enum):
    """Enum for representing the direction of the snake's movement."""
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()
