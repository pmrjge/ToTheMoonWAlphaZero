from multiprocessing.dummy import current_process
from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
import pax
from board import Board

from env import Enviroment
from utils import select_tree

class ChessGame(Enviroment):
    """Connect-Four game environment"""

    board: chex.Array
    who_play: chex.Array
    terminated: chex.Array
    winner: chex.Array
    num_cols: int
    num_rows: int

    def __init__(self, num_cols: int = 7, num_rows: int = 6):
        super().__init__()
        self.logic = Board()
        self.board = self.logic.current_board
        self.who_play = jnp.array(self.logic.player, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(1, dtype=jnp.int32)

    def num_actions(self):
        return 4672

    def invalid_actions(self) -> chex.Array:
        possible_moves = self.logic.actions()
        encoded = []
        for initial_pos, final_pos, underpromote in possible_moves:
            encoded.append(self.logic.encode_action(initial_pos, final_pos, underpromote)[0])
        invalid_moves = jnp.array(list(set(range(4672)).difference(set(encoded))), dtype=jnp.int32)

        result = jnp.zeros((4672), dtype=jnp.int32)
        return result.at[invalid_moves].set(1)

    def reset(self):
        self.logic = Board()
        self.board = self.logic.current_board
        self.who_play = jnp.array(self.logic.player, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(1, dtype=jnp.int32)

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["ChessGame", chex.Array]:
        """One step of the game.
        An invalid move will terminate the game with reward -1.
        """
        if self.invalid_moves()[action] == 1:
            return self, -10

        i_pos, f_pos, promote = self.logic.decode_action(action)
        self.logic.move_piece(i_pos, f_pos, promote)

        if self.logic.check_status() == True and self.logic.in_check_possible_moves() == []:
            if self.logic.player == 0:
                return self, -1.0
            elif self.logic.player == 1:
                return self, 1.0
        elif self.logic.in_check_possible_moves() == []:
            return self, 0.5
    
        return self, 0.0

    def render(self) -> None:
        """Render the game on screen."""
        print(self.logic.current_board)
        print()
        print()

    def observation(self) -> chex.Array:
        return self.logic.current_board

    def canonical_observation(self) -> chex.Array:
        return self.logic.current_board * self.who_play

    def is_terminated(self):
        return (self.logic.check_status() == True and self.logic.in_check_possible_moves() == []) or self.logic.in_check_possible_moves() == []

    def max_num_steps(self) -> int:
        return 500