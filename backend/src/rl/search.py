from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from cube.simulator import Cube3, Cube3State, MOVES


@dataclass
class SearchResult:
    solved: bool
    moves: List[str]
    cost: float
    nodes_generated: int
    path_states: List[np.ndarray]


class Node:
    __slots__ = [
        "state",
        "path_cost",
        "heuristic",
        "cost",
        "is_solved",
        "parent_move",
        "parent",
        "transition_costs",
        "children",
    ]

    def __init__(
        self,
        state: Cube3State,
        path_cost: float,
        is_solved: bool,
        parent_move: Optional[int],
        parent: Optional["Node"],
    ) -> None:
        self.state = state
        self.path_cost = path_cost
        self.heuristic: float = 0.0
        self.cost: float = 0.0
        self.is_solved = is_solved
        self.parent_move = parent_move
        self.parent = parent
        self.transition_costs: List[float] = []
        self.children: List["Node"] = []


def make_heuristic_fn(model: torch.nn.Module, device: torch.device, env: Cube3, batch_size: int = 32768) -> Callable[[List[Cube3State]], np.ndarray]:
    model.eval()
    use_cuda = device.type == "cuda"

    def heuristic(states: List[Cube3State]) -> np.ndarray:
        if not states:
            return np.zeros(0, dtype=np.float32)

        state_input = env.state_to_nnet_input(states)[0]
        outputs: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(state_input), batch_size):
                batch_np = state_input[start:start + batch_size]
                batch = torch.as_tensor(batch_np, device=device, non_blocking=use_cuda)
                values = model(batch).squeeze(-1)
                values = torch.clamp(values, min=0.0)
                outputs.append(values.cpu().numpy())
        heuristics = np.concatenate(outputs, axis=0)
        return heuristics

    return heuristic


def weighted_astar(
    start_state: Cube3State,
    env: Cube3,
    heuristic_fn: Callable[[List[Cube3State]], np.ndarray],
    weight: float = 0.8,
    batch_size: int = 1,
    max_expansions: int = 10_000,
) -> SearchResult:
    root = Node(start_state, 0.0, env.is_solved_single(start_state), None, None)
    root.heuristic = float(heuristic_fn([root.state])[0])
    root.cost = weight * root.path_cost + root.heuristic

    open_set: List[Tuple[float, int, Node]] = []
    closed_cost: dict[Cube3State, float] = {root.state: 0.0}
    counter = count()
    nodes_generated = 1

    heappush(open_set, (root.cost, next(counter), root))

    if root.is_solved:
        return SearchResult(True, [], 0.0, 1, [(root.state.colors // 9).astype(np.int8)])

    while open_set and nodes_generated <= max_expansions:
        nodes_to_expand: List[Node] = []
        while open_set and len(nodes_to_expand) < batch_size:
            nodes_to_expand.append(heappop(open_set)[2])

        states = [node.state for node in nodes_to_expand]
        children_states_by_node, transition_costs = env.expand(states)
        flat_children: List[Cube3State] = [child for children in children_states_by_node for child in children]
        flat_heuristics = heuristic_fn(flat_children)
        flat_is_solved = env.is_solved(flat_children)

        offset = 0
        for node_idx, parent in enumerate(nodes_to_expand):
            child_states = children_states_by_node[node_idx]
            child_tcs = transition_costs[node_idx]
            child_heuristics = flat_heuristics[offset:offset + len(child_states)]
            child_solved = flat_is_solved[offset:offset + len(child_states)]
            offset += len(child_states)

            for move_idx, child_state in enumerate(child_states):
                path_cost = parent.path_cost + float(child_tcs[move_idx])
                prev_best = closed_cost.get(child_state)
                if prev_best is not None and prev_best <= path_cost:
                    continue

                is_solved = bool(child_solved[move_idx])
                child = Node(child_state, path_cost, is_solved, move_idx, parent)
                child.heuristic = 0.0 if is_solved else float(child_heuristics[move_idx])
                child.cost = weight * child.path_cost + child.heuristic
                parent.children.append(child)
                parent.transition_costs.append(float(child_tcs[move_idx]))
                closed_cost[child.state] = path_cost
                nodes_generated += 1

                if is_solved:
                    return _build_result(child)

                heappush(open_set, (child.cost, next(counter), child))

    return SearchResult(False, [], float("inf"), nodes_generated, [])


def _build_result(goal_node: Node) -> SearchResult:
    moves: List[str] = []
    states: List[np.ndarray] = []
    current: Optional[Node] = goal_node
    while current is not None:
        states.append((current.state.colors // 9).astype(np.int8))
        if current.parent_move is not None:
            moves.append(MOVES[current.parent_move])
        current = current.parent

    states.reverse()
    moves.reverse()
    return SearchResult(True, moves, goal_node.path_cost, len(states), states)
