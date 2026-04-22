from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from cube.simulator import Cube3, Cube3State
from .network import get_device, maybe_compile
from .search import SearchResult, make_heuristic_fn, weighted_astar

LOGGER = logging.getLogger(__name__)


@dataclass
class AgentState:
    epoch: int = 0
    update_num: int = 0
    train_steps: int = 0
    scramble_depth: int = 1


class DeepCubeAgent:
    def __init__(
        self,
        hidden_size: int = 5000,
        learning_rate: float = 1e-3,
        lr_decay: float = 0.999999,
        artifact_root: str = "saved_models/cube3",
        search_weight: float = 0.8,
    ) -> None:
        self.env = Cube3()
        self.device = get_device()
        self.search_weight = search_weight
        self.artifact_root = self._resolve_artifact_root(artifact_root)
        self.current_dir = self.artifact_root / "current"
        self.target_dir = self.artifact_root / "target"
        self.checkpoint_dir = self.artifact_root / "checkpoints"
        self.model_config = {
            "hidden_size": hidden_size,
            "resnet_dim": max(hidden_size // 5, 128),
            "num_blocks": 4,
            "batch_norm": True,
        }
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_model = maybe_compile(self.env.get_nnet_model(hidden_size=hidden_size).to(self.device))
        self.target_model = maybe_compile(self.env.get_nnet_model(hidden_size=hidden_size).to(self.device))
        self.optimizer = torch.optim.Adam(self.current_model.parameters(), lr=learning_rate)
        self.lr_decay = lr_decay
        self.state = AgentState()

        self.use_amp = os.environ.get("USE_AMP", "0") == "1" and self.device.type == "cuda"
        self.scaler = GradScaler('cuda') if self.use_amp else None

        heuristic_batch = int(os.environ.get("HEURISTIC_BATCH_SIZE", "32768"))
        self.heuristic_current = make_heuristic_fn(self.current_model, self.device, self.env, batch_size=heuristic_batch)
        self.heuristic_target = make_heuristic_fn(self.target_model, self.device, self.env, batch_size=heuristic_batch)

    def bellman_backup(self, states: List[Cube3State], use_target: bool = True) -> np.ndarray:
        heuristic_fn = self.heuristic_target if use_target else self.heuristic_current
        children_by_state, transition_costs = self.env.expand(states)
        flat_children = [child for children in children_by_state for child in children]
        heuristics = heuristic_fn(flat_children)
        heuristics[self.env.is_solved(flat_children)] = 0.0

        tc_stack = np.stack(transition_costs)
        heuristics_mat = heuristics.reshape(len(states), -1)
        backups = np.min(heuristics_mat + tc_stack, axis=1)

        backups[self.env.is_solved(states)] = 0.0

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return backups

    def train_on_states(
        self,
        states: List[Cube3State],
        targets: np.ndarray,
        batch_size: int,
        num_iterations: int,
    ) -> List[float]:
        inputs = self.env.state_to_nnet_input(states)[0]
        inputs_t = torch.as_tensor(inputs, device=self.device)
        targets_t = torch.as_tensor(targets, device=self.device).unsqueeze(-1)

        loss_accum: List[torch.Tensor] = []
        self.current_model.train()
        for _ in range(num_iterations):
            idxs = np.random.choice(len(states), size=min(batch_size, len(states)), replace=False)
            batch_x = inputs_t[idxs]
            batch_y = targets_t[idxs]

            if self.use_amp:
                with autocast('cuda'):
                    preds = self.current_model(batch_x)
                    loss = torch.nn.functional.mse_loss(preds, batch_y)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), max_norm=1.0)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= self.lr_decay
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.current_model(batch_x)
                loss = torch.nn.functional.mse_loss(preds, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), max_norm=1.0)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= self.lr_decay
                self.optimizer.step()

            loss_accum.append(loss.detach())
            self.state.train_steps += 1

        losses = torch.stack(loss_accum).cpu().numpy().tolist() if loss_accum else []
        heuristic_batch = int(os.environ.get("HEURISTIC_BATCH_SIZE", "32768"))
        self.heuristic_current = make_heuristic_fn(self.current_model, self.device, self.env, batch_size=heuristic_batch)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return losses

    def promote_target(self) -> None:
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.state.update_num += 1
        self.heuristic_target = make_heuristic_fn(self.target_model, self.device, self.env)

    def solve(self, state: Cube3State, max_expansions: int = 10_000, batch_size: int = 1) -> SearchResult:
        result = weighted_astar(
            state,
            self.env,
            self.heuristic_current,
            weight=self.search_weight,
            batch_size=batch_size,
            max_expansions=max_expansions,
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def save_artifacts(self) -> None:
        torch.save(self.current_model.state_dict(), self.current_dir / "model_state_dict.pt")
        torch.save(self.target_model.state_dict(), self.target_dir / "model_state_dict.pt")
        torch.save(self.model_config, self.current_dir / "model_config.pt")
        torch.save(self.model_config, self.target_dir / "model_config.pt")
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "agent_state": asdict(self.state),
                "model_config": self.model_config,
            },
            self.current_dir / "trainer_state.pt",
        )
        torch.save(
            {
                "agent_state": asdict(self.state),
                "model_config": self.model_config,
            },
            self.target_dir / "trainer_state.pt",
        )

    def save_checkpoint(self, path: str) -> None:
        import shutil
        import tempfile
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        usage = shutil.disk_usage(path_obj.parent)
        if usage.free < 500 * 1024 * 1024:  # 500 MB headroom
            raise RuntimeError(
                f"Insufficient disk space to save checkpoint. "
                f"Free: {usage.free / 1e9:.2f} GB on {path_obj.parent}"
            )

        with tempfile.NamedTemporaryFile(dir=path_obj.parent, delete=False, suffix=".pt.tmp") as tmp:
            tmp_path = tmp.name
        try:
            torch.save(
                {
                    "current_model": self.current_model.state_dict(),
                    "target_model": self.target_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "agent_state": asdict(self.state),
                    "model_config": self.model_config,
                },
                tmp_path,
            )
            Path(tmp_path).replace(path_obj)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_config = checkpoint.get("model_config")
        if checkpoint_config and checkpoint_config != self.model_config:
            raise RuntimeError(
                f"Checkpoint model config {checkpoint_config} does not match current config {self.model_config}"
            )

        self._safe_load_model(self.current_model, checkpoint["current_model"], str(path), "current checkpoint")
        self._safe_load_model(self.target_model, checkpoint["target_model"], str(path), "target checkpoint")
        self._safe_load_optimizer(checkpoint.get("optimizer"), str(path))
        self.state = AgentState(**checkpoint["agent_state"])
        self.heuristic_current = make_heuristic_fn(self.current_model, self.device, self.env)
        self.heuristic_target = make_heuristic_fn(self.target_model, self.device, self.env)

    def _load_artifacts(self) -> None:
        current_model_path = self.current_dir / "model_state_dict.pt"
        target_model_path = self.target_dir / "model_state_dict.pt"
        current_state_path = self.current_dir / "trainer_state.pt"
        current_config_path = self.current_dir / "model_config.pt"
        target_config_path = self.target_dir / "model_config.pt"

        current_model_loaded = False
        target_model_loaded = False

        if current_model_path.exists():
            current_model_loaded = self._load_model_artifact(
                self.current_model,
                current_model_path,
                current_config_path,
                "current artifact",
            )
        if target_model_path.exists():
            target_model_loaded = self._load_model_artifact(
                self.target_model,
                target_model_path,
                target_config_path,
                "target artifact",
            )

        if not target_model_loaded:
            self.target_model.load_state_dict(self.current_model.state_dict())

        if current_state_path.exists():
            trainer_state: Dict[str, Any] = torch.load(current_state_path, map_location=self.device)
            state_config = trainer_state.get("model_config")
            if state_config and state_config != self.model_config:
                LOGGER.warning(
                    "Ignoring trainer state at %s because model config %s does not match current config %s",
                    current_state_path,
                    state_config,
                    self.model_config,
                )
                return
            if current_model_loaded and "optimizer" in trainer_state:
                self._safe_load_optimizer(trainer_state["optimizer"], str(current_state_path))
            if current_model_loaded and "agent_state" in trainer_state:
                self.state = AgentState(**trainer_state["agent_state"])

    def _load_model_artifact(
        self,
        model: torch.nn.Module,
        model_path: Path,
        config_path: Path,
        label: str,
    ) -> bool:
        artifact_config = self._load_model_config(config_path)
        if artifact_config and artifact_config != self.model_config:
            self._quarantine_artifact(model_path, f"{label}.config-mismatch")
            if config_path.exists():
                self._quarantine_artifact(config_path, f"{label}.config-mismatch")
            LOGGER.warning(
                "Skipped %s at %s due to model config mismatch; moved aside for a clean restart",
                label,
                model_path,
            )
            return False

        state_dict = torch.load(model_path, map_location=self.device)
        loaded = self._safe_load_model(model, state_dict, str(model_path), label)
        if not loaded:
            self._quarantine_artifact(model_path, f"{label}.incompatible")
            if config_path.exists():
                self._quarantine_artifact(config_path, f"{label}.incompatible")
            LOGGER.warning(
                "Skipped %s at %s because it does not match the current network; moved aside for a clean restart",
                label,
                model_path,
            )
        return loaded

    def _load_model_config(self, config_path: Path) -> Dict[str, Any] | None:
        if not config_path.exists():
            return None
        config = torch.load(config_path, map_location="cpu")
        if isinstance(config, dict):
            return config
        LOGGER.warning("Ignoring malformed model config at %s", config_path)
        return None

    def _safe_load_model(
        self,
        model: torch.nn.Module,
        state_dict: Dict[str, Any],
        source: str,
        label: str,
    ) -> bool:
        try:
            model.load_state_dict(state_dict)
            return True
        except RuntimeError:
            return False

    def _safe_load_optimizer(self, optimizer_state: Dict[str, Any] | None, source: str) -> bool:
        if optimizer_state is None:
            return False
        try:
            self.optimizer.load_state_dict(optimizer_state)
            return True
        except ValueError:
            LOGGER.warning("Ignoring incompatible optimizer state from %s", source)
            return False

    def _resolve_artifact_root(self, artifact_root: str) -> Path:
        path = Path(artifact_root)
        if path.is_absolute():
            return path
        backend_root = Path(__file__).resolve().parents[2]
        return backend_root / path

    def _quarantine_artifact(self, path: Path, reason: str) -> Path | None:
        if not path.exists():
            return None

        target = path.with_name(f"{path.stem}.{reason}{path.suffix}")
        counter = 1
        while target.exists():
            target = path.with_name(f"{path.stem}.{reason}.{counter}{path.suffix}")
            counter += 1

        path.rename(target)
        return target
