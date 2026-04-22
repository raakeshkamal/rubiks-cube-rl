from __future__ import annotations

from dataclasses import dataclass
from random import choice
from typing import Dict, List, Tuple, Union

import numpy as np

from .environment import Environment, State


MOVES = ["U'", "U", "D'", "D", "L'", "L", "R'", "R", "B'", "B", "F'", "F"]
FACE_SIZE = 9
TOTAL_SIZE = 54
SOLVED_STATE = np.repeat(np.arange(6, dtype=np.int8), FACE_SIZE)


def reverse_moves(moves: List[str]) -> List[str]:
    return [move[0] if "'" in move else f"{move}'" for move in reversed(moves)]


def _to_face_colors(sticker_state: np.ndarray) -> np.ndarray:
    return (sticker_state // FACE_SIZE).astype(np.int8)


@dataclass(eq=False, slots=True)
class Cube3State(State):
    colors: np.ndarray
    hash_value: int | None = None

    def __hash__(self) -> int:
        if self.hash_value is None:
            self.hash_value = hash(self.colors.tobytes())
        return self.hash_value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cube3State):
            return False
        if self.hash_value is not None and other.hash_value is not None and self.hash_value != other.hash_value:
            return False
        return np.array_equal(self.colors, other.colors)

    def copy(self) -> "Cube3State":
        return Cube3State(self.colors.copy())


class Cube3(Environment):
    def __init__(self) -> None:
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 3
        self.goal_colors = np.arange(0, (self.cube_len ** 2) * 6, dtype=self.dtype)

        self.adj_faces: Dict[int, np.ndarray]
        self._get_adj()
        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(self.cube_len, MOVES)

    def next_state(self, states: List[Cube3State], action: int) -> Tuple[List[Cube3State], List[float]]:
        states_np = np.stack([state.colors for state in states], axis=0)
        states_next_np, transition_costs = self._move_np(states_np, action)
        return [Cube3State(state) for state in states_next_np], transition_costs

    def prev_state(self, states: List[Cube3State], action: int) -> List[Cube3State]:
        move = MOVES[action]
        move_rev_idx = MOVES.index(reverse_moves([move])[0])
        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[Cube3State], np.ndarray]:
        if np_format:
            goal_np = np.expand_dims(self.goal_colors.copy(), 0)
            return np.repeat(goal_np, num_states, axis=0)
        return [Cube3State(self.goal_colors.copy()) for _ in range(num_states)]

    def is_solved(self, states: List[Cube3State]) -> np.ndarray:
        states_np = np.stack([state.colors for state in states], axis=0)
        is_equal = np.equal(states_np, np.expand_dims(self.goal_colors, 0))
        return np.all(is_equal, axis=1)

    def is_solved_single(self, state: Cube3State) -> bool:
        return np.array_equal(state.colors, self.goal_colors)

    def state_to_nnet_input(self, states: List[Cube3State]) -> List[np.ndarray]:
        states_np = np.stack([state.colors for state in states], axis=0)
        return [(states_np // FACE_SIZE).astype(self.dtype)]

    def get_num_moves(self) -> int:
        return len(MOVES)

    def get_nnet_model(self, hidden_size: int = 5000):
        from rl.network import DeepCubeNet

        resnet_dim = max(hidden_size // 5, 128)
        return DeepCubeNet(h1_dim=hidden_size, resnet_dim=resnet_dim, num_blocks=4, batch_norm=True)

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[Cube3State], List[int]]:
        assert num_states > 0
        assert backwards_range[0] >= 0

        scrambles = list(range(backwards_range[0], backwards_range[1] + 1))
        num_moves = self.get_num_moves()
        states_np = self.generate_goal_states(num_states, np_format=True)

        scramble_nums = np.random.choice(scrambles, num_states)
        num_back_moves = np.zeros(num_states, dtype=np.int32)

        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs = np.where(moves_lt)[0]
            subset_size = int(max(len(idxs) / num_moves, 1))
            idxs = np.random.choice(idxs, subset_size)

            move = np.random.randint(0, num_moves)
            states_np[idxs], _ = self._move_np(states_np[idxs], move)

            num_back_moves[idxs] += 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        return [Cube3State(state) for state in states_np], scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        num_states = len(states)
        num_moves = self.get_num_moves()
        states_exp: List[List[State]] = [[] for _ in range(num_states)]
        tc = np.empty([num_states, num_moves], dtype=np.float32)

        states_np = np.stack([state.colors for state in states])

        for move_idx in range(num_moves):
            states_next_np, tc_move = self._move_np(states_np, move_idx)
            tc[:, move_idx] = np.asarray(tc_move, dtype=np.float32)
            for idx in range(num_states):
                states_exp[idx].append(Cube3State(states_next_np[idx]))

        return states_exp, [tc[idx] for idx in range(num_states)]

    def _move_np(self, states_np: np.ndarray, action: int) -> Tuple[np.ndarray, List[float]]:
        action_str = MOVES[action]
        states_next_np = states_np.copy()
        states_next_np[:, self.rotate_idxs_new[action_str]] = states_np[:, self.rotate_idxs_old[action_str]]
        return states_next_np, [1.0 for _ in range(states_np.shape[0])]

    def _get_adj(self) -> None:
        self.adj_faces = {
            0: np.array([2, 5, 3, 4]),
            1: np.array([2, 4, 3, 5]),
            2: np.array([0, 4, 1, 5]),
            3: np.array([0, 5, 1, 4]),
            4: np.array([0, 3, 1, 2]),
            5: np.array([0, 2, 1, 3]),
        }

    def _compute_rotation_idxs(self, cube_len: int, moves: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        rotate_idxs_new: Dict[str, np.ndarray] = {}
        rotate_idxs_old: Dict[str, np.ndarray] = {}
        face_dict = {"U": 0, "D": 1, "L": 2, "R": 3, "B": 4, "F": 5}

        for move in moves:
            face = face_dict[move[0]]
            sign = -1 if "'" in move else 1

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors = np.zeros((6, cube_len, cube_len), dtype=np.int64)
            colors_new = np.copy(colors)

            adj_idxs = {
                0: {2: [range(0, cube_len), cube_len - 1], 3: [range(0, cube_len), cube_len - 1],
                    4: [range(0, cube_len), cube_len - 1], 5: [range(0, cube_len), cube_len - 1]},
                1: {2: [range(0, cube_len), 0], 3: [range(0, cube_len), 0],
                    4: [range(0, cube_len), 0], 5: [range(0, cube_len), 0]},
                2: {0: [0, range(0, cube_len)], 1: [0, range(0, cube_len)],
                    4: [cube_len - 1, range(cube_len - 1, -1, -1)], 5: [0, range(0, cube_len)]},
                3: {0: [cube_len - 1, range(0, cube_len)], 1: [cube_len - 1, range(0, cube_len)],
                    4: [0, range(cube_len - 1, -1, -1)], 5: [cube_len - 1, range(0, cube_len)]},
                4: {0: [range(0, cube_len), cube_len - 1], 1: [range(cube_len - 1, -1, -1), 0],
                    2: [0, range(0, cube_len)], 3: [cube_len - 1, range(cube_len - 1, -1, -1)]},
                5: {0: [range(0, cube_len), 0], 1: [range(cube_len - 1, -1, -1), cube_len - 1],
                    2: [cube_len - 1, range(0, cube_len)], 3: [0, range(cube_len - 1, -1, -1)]},
            }

            faces_to = self.adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[(np.arange(len(faces_to) - 1, len(faces_to) - 1 + len(faces_to))) % len(faces_to)]

            cubes_idxs = [
                [0, range(0, cube_len)],
                [range(0, cube_len), cube_len - 1],
                [cube_len - 1, range(cube_len - 1, -1, -1)],
                [range(cube_len - 1, -1, -1), 0],
            ]
            cubes_to = np.array([0, 1, 2, 3])
            if sign == 1:
                cubes_from = cubes_to[(np.arange(len(cubes_to) - 1, len(cubes_to) - 1 + len(cubes_to))) % len(cubes_to)]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for idx in range(4):
                idxs_new = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_to[idx]][0]]).flatten()
                            for idx2 in np.array([cubes_idxs[cubes_to[idx]][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_from[idx]][0]]).flatten()
                            for idx2 in np.array([cubes_idxs[cubes_from[idx]][1]]).flatten()]
                for idx_new, idx_old in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face, idx_new[0], idx_new[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face, idx_old[0], idx_old[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

            face_idxs = adj_idxs[face]
            for idx in range(len(faces_to)):
                face_to = faces_to[idx]
                face_from = faces_from[idx]
                idxs_new = [[idx1, idx2] for idx1 in np.array([face_idxs[face_to][0]]).flatten()
                            for idx2 in np.array([face_idxs[face_to][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([face_idxs[face_from][0]]).flatten()
                            for idx2 in np.array([face_idxs[face_from][1]]).flatten()]
                for idx_new, idx_old in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face_to, idx_new[0], idx_new[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face_from, idx_old[0], idx_old[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

        return rotate_idxs_new, rotate_idxs_old


class CubeSimulator:
    def __init__(self) -> None:
        self.env = Cube3()
        self._state = self.env.generate_goal_states(1)[0]

    @property
    def state(self) -> np.ndarray:
        return _to_face_colors(self._state.colors)

    @property
    def sticker_state(self) -> np.ndarray:
        return self._state.colors.copy()

    def reset(self) -> np.ndarray:
        self._state = self.env.generate_goal_states(1)[0]
        return self.state.copy()

    def is_solved(self) -> bool:
        return bool(self.env.is_solved([self._state])[0])

    def apply_move(self, move: str) -> np.ndarray:
        action = MOVES.index(move)
        self._state = self.env.next_state([self._state], action)[0][0]
        return self.state.copy()

    def apply_action(self, action: int) -> np.ndarray:
        self._state = self.env.next_state([self._state], action)[0][0]
        return self.state.copy()

    def scramble(self, n: int = 20) -> Tuple[np.ndarray, List[str]]:
        self.reset()
        moves: List[str] = []
        for _ in range(n):
            move = choice(MOVES)
            self.apply_move(move)
            moves.append(move)
        return self.state.copy(), moves

    def distance_to_solved(self) -> int:
        return int(np.count_nonzero(self.state != SOLVED_STATE))

    def apply_moves(self, moves: List[str]) -> None:
        for move in moves:
            self.apply_move(move)
