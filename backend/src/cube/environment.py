from __future__ import annotations

from abc import ABC, abstractmethod
from random import randrange
from typing import Any, List, Tuple

import numpy as np


class State(ABC):
    __slots__ = ()

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError


class Environment(ABC):
    def __init__(self) -> None:
        self.dtype = np.float32
        self.fixed_actions = True

    @abstractmethod
    def next_state(self, states: List[State], action: int) -> Tuple[List[State], List[float]]:
        raise NotImplementedError

    @abstractmethod
    def prev_state(self, states: List[State], action: int) -> List[State]:
        raise NotImplementedError

    @abstractmethod
    def generate_goal_states(self, num_states: int, np_format: bool = False):
        raise NotImplementedError

    @abstractmethod
    def is_solved(self, states: List[State]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def state_to_nnet_input(self, states: List[State]) -> List[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def get_num_moves(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_nnet_model(self, hidden_size: int = 5000) -> Any:
        raise NotImplementedError

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[State], List[int]]:
        assert num_states > 0
        assert backwards_range[0] >= 0
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        scrambles = list(range(backwards_range[0], backwards_range[1] + 1))
        num_moves = self.get_num_moves()
        states = self.generate_goal_states(num_states)

        scramble_nums = np.random.choice(scrambles, num_states)
        num_back_moves = np.zeros(num_states, dtype=np.int32)

        while np.any(num_back_moves < scramble_nums):
            idxs = np.where(num_back_moves < scramble_nums)[0]
            subset_size = int(max(len(idxs) / num_moves, 1))
            idxs = np.random.choice(idxs, subset_size)

            move = randrange(num_moves)
            moved_states = self.prev_state([states[idx] for idx in idxs], move)
            for dest_idx, moved_state in zip(idxs, moved_states):
                states[dest_idx] = moved_state

            num_back_moves[idxs] += 1

        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        num_states = len(states)
        num_moves = self.get_num_moves()
        states_exp: List[List[State]] = [[] for _ in range(num_states)]
        transition_costs = np.empty([num_states, num_moves], dtype=np.float32)

        for move_idx in range(num_moves):
            states_next, tc_move = self.next_state(states, move_idx)
            transition_costs[:, move_idx] = np.asarray(tc_move, dtype=np.float32)
            for state_idx in range(num_states):
                states_exp[state_idx].append(states_next[state_idx])

        return states_exp, [transition_costs[idx] for idx in range(num_states)]
