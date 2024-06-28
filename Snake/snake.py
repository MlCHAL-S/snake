import pygame as pg
from random import randint
from .constants import BLOCK_SIZE, SPEED, Point
from .direction import Direction
from typing import List, Tuple, Optional
import numpy as np

pg.init()


class SnakeGame:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        """
        Initialize the Snake game.

        Args:
            width (int): The width of the game window.
            height (int): The height of the game window.
        """
        self.width: int = width
        self.height: int = height
        self.display: pg.Surface = pg.display.set_mode((self.width, self.height))
        self.clock: pg.time.Clock = pg.time.Clock()
        self.direction: Direction = Direction.RIGHT
        self.font = pg.font.SysFont('calibri', 25)

        pg.display.set_caption('Snake')

        self.head: Point = Point(self.width // 2, self.height // 2)
        self.snake: List[Point] = [self.head,
                                   Point(self.head.x - BLOCK_SIZE, self.head.y),
                                   Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score: int = 0
        self.food: Optional[Point] = None
        self._place_food()
        self.frame_iteration = 0

    def reset(self) -> None:
        """
        Resets the game
        """
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self) -> None:
        """Place food randomly on the game board."""
        x: int = randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y: int = randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action) -> Tuple[int, bool, int]:
        """
        Execute one step of the game.

        Returns:
            Tuple[bool, int]: A tuple containing a boolean indicating game over status and the current score.
        """
        reward: int = 0
        self.frame_iteration += 1

        # 1. collect user input
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = -10
            return reward, True, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_board()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, False, self.score

    def _update_board(self) -> None:
        """Update game board."""
        self.display.fill(pg.Color(0, 0, 0))

        for point in self.snake:
            pg.draw.rect(self.display, pg.Color(0, 100, 0), pg.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pg.draw.rect(self.display, pg.Color(0, 255, 0), pg.Rect(point.x + 4, point.y + 4, 12, 12))

        pg.draw.rect(self.display, pg.Color(100, 0, 0), pg.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pg.draw.rect(self.display, pg.Color(255, 0, 0), pg.Rect(self.food.x + 4, self.food.y + 4, 12, 12))

        text = self.font.render(f'Score: {self.score}', True, pg.Color(255, 255, 255))
        self.display.blit(text, (0, 0))
        pg.display.flip()

    def _move(self, action) -> None:
        """
        Move the snake in the given direction. [straight, right, left]

        Args:
            action (###): The direction to move the snake.
        """
        clock_wise: List = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_direction_index: int = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[current_direction_index]
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (current_direction_index + 1) % 4
            new_direction = clock_wise[next_index]
        else:  # [0, 0, 1]
            next_index = (current_direction_index - 1) % 4
            new_direction = clock_wise[next_index]

        self.direction = new_direction

        x: int = self.head.x
        y: int = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def is_collision(self, point: Point = None) -> bool:
        if point is None:
            point = self.head
        """
        Check if the snake has collided with the walls or itself.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        # Check collision with boundaries
        if (point.x >= self.width or point.x < 0 or
                point.y >= self.height or point.y < 0):
            return True

        # Check collision with itself
        if point in self.snake[1:]:
            return True

        return False
