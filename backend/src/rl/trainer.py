from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .agent import DeepCubeAgent


@dataclass
class TrainingConfig:
    scramble_depth: int = 1
    max_scramble_depth: int = 20
    episodes_per_epoch: int = 32
    checkpoint_freq: int = 5
    hidden_size: int = 5000
    learning_rate: float = 1e-3
    batch_size: int = 10000
    states_per_update: int = 262_144
    train_epochs_per_update: int = 100
    loss_thresh: float = 0.05
    back_max: int = 20
    max_steps_per_episode: int = 30
    max_search_nodes: int = 20_000
    eval_batch_size: int = 32
    search_weight: float = 0.8
    artifact_root: str = "saved_models/cube3"
    max_epochs: int = 10_000
    keep_last_n_checkpoints: int = 10

    # Compatibility fields kept for the existing UI payload shape.
    epsilon_start: float = 0.0
    epsilon_end: float = 0.0
    epsilon_decay: float = 1.0
    adi_states_per_step: int = 512
    adi_steps_per_epoch: int = 8
    target_update_freq: int = 1
    loss_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.states_per_update <= 0:
            self.states_per_update = max(self.adi_states_per_step * self.adi_steps_per_epoch, 1024)
        if self.loss_thresh <= 0:
            self.loss_thresh = self.loss_threshold or 0.1
        if self.back_max < self.scramble_depth:
            self.back_max = self.scramble_depth


@dataclass
class TrainingMetrics:
    episode: int = 0
    total_steps: int = 0
    epoch: int = 0
    solved: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    solve_rate: float = 0.0
    epsilon: float = 0.0
    avg_loss: float = 0.0
    epoch_time: float = 0.0
    episodes_per_sec: float = 0.0
    current_scramble_depth: int = 1
    device: str = "cpu"
    adi_steps: int = 0
    lr: float = 1e-3
    update_num: int = 0


class Trainer:
    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()
        self.agent = DeepCubeAgent(
            hidden_size=self.config.hidden_size,
            learning_rate=self.config.learning_rate,
            artifact_root=self.config.artifact_root,
            search_weight=self.config.search_weight,
        )
        self.metrics = TrainingMetrics(
            device=str(self.agent.device),
            current_scramble_depth=max(self.config.scramble_depth, self.agent.state.scramble_depth),
            lr=self.config.learning_rate,
            epoch=self.agent.state.epoch,
            update_num=self.agent.state.update_num,
            total_steps=self.agent.state.train_steps,
        )
        self.config.scramble_depth = self.metrics.current_scramble_depth
        self.is_running = False
        self.should_stop = False
        self.stop_reason: Optional[str] = None

    async def run_epoch(
        self,
        on_episode: Optional[Callable[[dict], None]] = None,
        on_checkpoint: Optional[Callable[[str], None]] = None,
        on_step: Optional[Callable[[np.ndarray, str], None]] = None,
        on_adi_progress: Optional[Callable[[dict], None]] = None,
        on_eval_step: Optional[Callable[[np.ndarray, str, int, bool, int], None]] = None,
    ) -> TrainingMetrics:
        del on_step
        self.is_running = True
        loop = asyncio.get_event_loop()
        start_time = time.time()

        states, _ = await loop.run_in_executor(
            None,
            self.agent.env.generate_states,
            self.config.states_per_update,
            (0, self.config.scramble_depth),
        )
        solved_state = self.agent.env.generate_goal_states(1)[0]
        states = [solved_state] + states
        targets = await loop.run_in_executor(None, self.agent.bellman_backup, states)

        num_train_iterations = max(
            1,
            int(np.ceil(len(states) / self.config.batch_size)) * self.config.train_epochs_per_update,
        )
        losses = await loop.run_in_executor(
            None,
            self.agent.train_on_states,
            states,
            targets,
            self.config.batch_size,
            num_train_iterations,
        )
        self.metrics.adi_steps += len(losses)
        self.metrics.total_steps = self.agent.state.train_steps

        if on_adi_progress:
            on_adi_progress(
                {
                    "adi_step": len(losses),
                    "adi_steps_total": len(losses),
                    "avg_loss": round(float(np.mean(losses)), 6),
                    "scramble_depth": self.config.scramble_depth,
                    "total_adi_steps": self.metrics.adi_steps,
                }
            )

        avg_loss = float(np.mean(losses))
        if avg_loss < self.config.loss_thresh:
            self.agent.promote_target()
            if self.config.scramble_depth < self.config.max_scramble_depth:
                self.config.scramble_depth += 1

        self.agent.state.epoch += 1
        self.agent.state.scramble_depth = self.config.scramble_depth

        eval_states, _ = await loop.run_in_executor(
            None,
            self.agent.env.generate_states,
            self.config.episodes_per_epoch,
            (self.config.scramble_depth, self.config.scramble_depth),
        )

        episode_results = []
        for eval_state in eval_states:
            if self.should_stop:
                break

            result = await loop.run_in_executor(
                None,
                self.agent.solve,
                eval_state,
                self.config.max_search_nodes,
                self.config.eval_batch_size,
            )
            episode_results.append(result)

            if on_eval_step and result.path_states:
                solved_face_state = np.repeat(np.arange(6), 9)
                for step_idx, state in enumerate(result.path_states[1:], start=1):
                    move = result.moves[step_idx - 1]
                    await on_eval_step(
                        state,
                        move,
                        step_idx,
                        step_idx == len(result.moves) and result.solved,
                        int(np.count_nonzero(state != solved_face_state)),
                    )

            self.metrics.episode += 1
            reward = 1.0 if result.solved else 0.0
            if on_episode:
                on_episode(
                    {
                        "episode": self.metrics.episode,
                        "reward": reward,
                        "solved": result.solved,
                        "steps": len(result.moves),
                        "scramble_depth": self.config.scramble_depth,
                    }
                )

        solved_count = sum(1 for result in episode_results if result.solved)
        total_reward = float(solved_count)
        elapsed = time.time() - start_time

        self.metrics.solved += solved_count
        self.metrics.total_reward += total_reward
        self.metrics.avg_reward = total_reward / max(len(episode_results), 1)
        self.metrics.solve_rate = solved_count / max(len(episode_results), 1)
        self.metrics.avg_loss = round(avg_loss, 6)
        self.metrics.epoch = self.agent.state.epoch
        self.metrics.epoch_time = elapsed
        self.metrics.episodes_per_sec = len(episode_results) / elapsed if elapsed > 0 else 0.0
        self.metrics.current_scramble_depth = self.config.scramble_depth
        self.metrics.lr = self.agent.optimizer.param_groups[0]["lr"]
        self.metrics.update_num = self.agent.state.update_num

        if self.metrics.epoch % self.config.checkpoint_freq == 0:
            checkpoint_path = Path(self.config.artifact_root) / "checkpoints" / f"epoch_{self.metrics.epoch:05d}.pt"
            await loop.run_in_executor(None, self.agent.save_checkpoint, str(checkpoint_path))
            if on_checkpoint:
                on_checkpoint(str(checkpoint_path))
            self._cleanup_old_checkpoints()

        if self.metrics.epoch >= self.config.max_epochs:
            self.should_stop = True
            self.stop_reason = f"Max epochs ({self.config.max_epochs}) reached"

        self.is_running = False
        return self.metrics

    def get_metrics_dict(self) -> dict:
        return asdict(self.metrics)

    def save_checkpoint(self, path: str) -> None:
        self.agent.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        self.agent.load_checkpoint(path)
        self.metrics.epoch = self.agent.state.epoch
        self.metrics.total_steps = self.agent.state.train_steps
        self.metrics.current_scramble_depth = self.agent.state.scramble_depth
        self.metrics.update_num = self.agent.state.update_num
        self.config.scramble_depth = self.agent.state.scramble_depth

    def stop(self) -> None:
        self.should_stop = True

    def reset(self) -> None:
        self.should_stop = False
        self.is_running = False

    def _cleanup_old_checkpoints(self) -> None:
        checkpoint_dir = Path(self.config.artifact_root) / "checkpoints"
        if not checkpoint_dir.exists():
            return
        checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        to_remove = checkpoints[:-self.config.keep_last_n_checkpoints]
        for cp in to_remove:
            cp.unlink(missing_ok=True)
