from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils import EpisodeBatch, make_synthetic_episodes


STATE_PREFIXES = ("state_", "obs_", "observation_", "vital_", "lab_", "price_", "indicator_")
ACTION_PREFIXES = ("action_", "dose_", "med_", "trade_", "position_")
NEXT_STATE_PREFIXES = ("next_state_", "next_obs_", "next_observation_")


class TrajectoryDataset(Dataset[EpisodeBatch]):
    """Dataset whose atomic item is a full correlated trajectory."""

    def __init__(self, episodes: Sequence[EpisodeBatch], shard_count: int = 1) -> None:
        if not episodes:
            raise ValueError("TrajectoryDataset requires at least one episode")
        if shard_count <= 0:
            raise ValueError("shard_count must be positive")
        self._episodes = list(episodes)
        self.shard_count = shard_count
        self._episode_index = {episode.episode_id: index for index, episode in enumerate(self._episodes)}
        if len(self._episode_index) != len(self._episodes):
            raise ValueError("episode_id values must be unique")
        self._validate_boundaries()
        self._shard_map = self._build_shard_map(shard_count)

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> EpisodeBatch:
        return self._episodes[index]

    @property
    def episode_ids(self) -> list[int]:
        return [episode.episode_id for episode in self._episodes]

    def get_episode(self, episode_id: int) -> EpisodeBatch:
        return self._episodes[self._episode_index[episode_id]]

    def without_episode(self, episode_id: int) -> "TrajectoryDataset":
        return TrajectoryDataset(
            [episode for episode in self._episodes if episode.episode_id != episode_id],
            shard_count=self.shard_count,
        )

    def shard_ids(self) -> list[int]:
        return sorted(self._shard_map)

    def shard_episode_ids(self, shard_id: int) -> tuple[int, ...]:
        if shard_id not in self._shard_map:
            raise KeyError(f"unknown shard_id {shard_id}")
        return tuple(self._shard_map[shard_id])

    def shard_dataset(self, shard_id: int) -> "TrajectoryDataset":
        return TrajectoryDataset(
            [self.get_episode(episode_id) for episode_id in self.shard_episode_ids(shard_id)],
            shard_count=1,
        )

    def affected_shards(self, deleted_episode_ids: Sequence[int]) -> set[int]:
        return {self.shard_for_episode(episode_id) for episode_id in deleted_episode_ids}

    def shard_for_episode(self, episode_id: int) -> int:
        if episode_id not in self._episode_index:
            raise KeyError(f"unknown episode_id {episode_id}")
        return episode_id % self.shard_count

    def split_shards(self, shard_count: int | None = None) -> list["TrajectoryDataset"]:
        if shard_count is not None and shard_count != self.shard_count:
            return TrajectoryDataset(self._episodes, shard_count=shard_count).split_shards()
        return [self.shard_dataset(shard_id) for shard_id in self.shard_ids()]

    def _validate_boundaries(self) -> None:
        for episode in self._episodes:
            length = episode.states.shape[0]
            tensors = (
                episode.actions,
                episode.rewards,
                episode.next_states,
                episode.dones,
            )
            if any(tensor.shape[0] != length for tensor in tensors):
                raise ValueError(f"episode {episode.episode_id} has inconsistent trajectory length")
            if episode.dones.ndim != 1:
                raise ValueError(f"episode {episode.episode_id} dones must be a 1D tensor")
            terminal_value = episode.dones.new_tensor(1.0)
            if episode.dones.numel() > 0 and not torch.isclose(episode.dones[-1], terminal_value):
                raise ValueError(f"episode {episode.episode_id} must terminate at the last transition")

    def _build_shard_map(self, shard_count: int) -> dict[int, list[int]]:
        shard_map = {shard_id: [] for shard_id in range(shard_count)}
        for episode_id in self.episode_ids:
            shard_map[episode_id % shard_count].append(episode_id)
        return shard_map

    @classmethod
    def from_transition_tensors(
        cls,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        episode_ids: Tensor,
    ) -> "TrajectoryDataset":
        episodes: list[EpisodeBatch] = []
        for episode_id in torch.unique(episode_ids, sorted=True):
            mask = episode_ids == episode_id
            episodes.append(
                EpisodeBatch(
                    episode_id=int(episode_id.item()),
                    states=states[mask],
                    actions=actions[mask],
                    rewards=rewards[mask],
                    next_states=next_states[mask],
                    dones=dones[mask],
                )
            )
        return cls(episodes)

    @classmethod
    def synthetic(
        cls,
        episode_count: int,
        horizon: int,
        state_dim: int,
        action_dim: int,
        generator: torch.Generator,
        shard_count: int = 1,
    ) -> "TrajectoryDataset":
        episodes = make_synthetic_episodes(
            episode_count=episode_count,
            horizon=horizon,
            state_dim=state_dim,
            action_dim=action_dim,
            generator=generator,
        )
        return cls(episodes, shard_count=shard_count)


class TrajectoryDataLoader:
    """Episode-level mini-batch loader using sampling without replacement."""

    def __init__(
        self,
        dataset: TrajectoryDataset,
        batch_size: int,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterator[list[EpisodeBatch]]:
        indices = torch.arange(len(self.dataset))
        if self.shuffle:
            indices = indices[torch.randperm(len(indices), generator=self.generator)]
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            yield [self.dataset[int(index)] for index in batch_indices]

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class RealWorldTrajectoryLoader:
    """Load medical or financial logs into episode-level `EpisodeBatch` objects."""

    def __init__(
        self,
        domain: str,
        data_path: str | Path | None,
        state_dim: int,
        action_dim: int,
        horizon: int,
        episode_count: int,
        shard_count: int,
        generator: torch.Generator,
    ) -> None:
        if domain not in {"medical", "financial"}:
            raise ValueError("domain must be 'medical' or 'financial'")
        self.domain = domain
        self.data_path = None if data_path is None else Path(data_path)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.episode_count = episode_count
        self.shard_count = shard_count
        self.generator = generator

    def load(self) -> TrajectoryDataset:
        if self.data_path is not None and self.data_path.exists():
            return self.from_csv(self.data_path)
        if self.domain == "medical":
            episodes = self.medical_proxy_episodes()
        else:
            episodes = self.financial_proxy_episodes()
        return TrajectoryDataset(episodes, shard_count=self.shard_count)

    def from_csv(self, path: Path) -> TrajectoryDataset:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{path} contains no rows")
        state_columns = self._columns(rows[0], STATE_PREFIXES, fallback_count=self.state_dim)
        action_columns = self._columns(rows[0], ACTION_PREFIXES, fallback_count=self.action_dim)
        next_state_columns = self._columns(rows[0], NEXT_STATE_PREFIXES, fallback_count=0)
        episode_key = self._first_present(rows[0], ("episode_id", "patient_id", "stay_id", "ticker", "asset_id"))
        time_key = self._first_present(rows[0], ("t", "time", "timestamp", "date", "step"), required=False)
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row[episode_key], []).append(row)
        episodes: list[EpisodeBatch] = []
        for episode_index, (_episode_key, episode_rows) in enumerate(sorted(grouped.items())):
            if time_key is not None:
                episode_rows = sorted(episode_rows, key=lambda item: item[time_key])
            states = self._tensor_from_columns(episode_rows, state_columns)
            actions = self._tensor_from_columns(episode_rows, action_columns)
            rewards = self._rewards_from_rows(episode_rows, actions)
            if next_state_columns:
                next_states = self._tensor_from_columns(episode_rows, next_state_columns)
            else:
                next_states = torch.cat((states[1:], states[-1:].clone()), dim=0)
            dones = torch.zeros(states.shape[0])
            dones[-1] = 1.0
            episodes.append(
                EpisodeBatch(
                    episode_id=episode_index,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones,
                )
            )
        return TrajectoryDataset(episodes, shard_count=self.shard_count)

    def medical_proxy_episodes(self) -> list[EpisodeBatch]:
        episodes: list[EpisodeBatch] = []
        treatment_matrix = torch.randn(
            self.state_dim,
            self.action_dim,
            generator=self.generator,
        ) / max(self.state_dim, 1) ** 0.5
        for episode_id in range(self.episode_count):
            severity = torch.randn(1, self.state_dim, generator=self.generator)
            states: list[Tensor] = []
            actions: list[Tensor] = []
            rewards: list[Tensor] = []
            for step in range(self.horizon):
                circadian = torch.sin(torch.linspace(0.0, 3.14, self.state_dim) + 0.1 * step)
                observation = severity + 0.15 * circadian + 0.08 * torch.randn(
                    1,
                    self.state_dim,
                    generator=self.generator,
                )
                action = torch.tanh(observation @ treatment_matrix)
                action = torch.clamp(
                    action + 0.10 * torch.randn(1, self.action_dim, generator=self.generator),
                    -1.0,
                    1.0,
                )
                burden = observation.square().mean(dim=-1)
                intervention_cost = 0.08 * action.square().sum(dim=-1)
                reward = 1.2 - burden - intervention_cost
                severity = 0.82 * severity - 0.10 * action.mean(dim=-1, keepdim=True) + 0.12 * torch.randn(
                    1,
                    self.state_dim,
                    generator=self.generator,
                )
                states.append(observation.squeeze(0))
                actions.append(action.squeeze(0))
                rewards.append(reward.squeeze(0))
            state_tensor = torch.stack(states)
            action_tensor = torch.stack(actions)
            reward_tensor = torch.stack(rewards)
            next_states = torch.cat((state_tensor[1:], state_tensor[-1:].clone()), dim=0)
            dones = torch.zeros(self.horizon)
            dones[-1] = 1.0
            episodes.append(EpisodeBatch(episode_id, state_tensor, action_tensor, reward_tensor, next_states, dones))
        return episodes

    def financial_proxy_episodes(self) -> list[EpisodeBatch]:
        episodes: list[EpisodeBatch] = []
        action_projection = torch.randn(
            self.state_dim,
            self.action_dim,
            generator=self.generator,
        ) / max(self.state_dim, 1) ** 0.5
        for episode_id in range(self.episode_count):
            market = torch.randn(self.horizon + 1, self.state_dim, generator=self.generator)
            seasonal = torch.sin(torch.linspace(0.0, 6.28, self.horizon + 1)).unsqueeze(-1)
            prices = torch.cumsum(0.04 * market + 0.03 * seasonal, dim=0)
            states = torch.cat((prices[:-1], torch.tanh(prices[:-1])), dim=-1)[:, : self.state_dim]
            next_states = torch.cat((prices[1:], torch.tanh(prices[1:])), dim=-1)[:, : self.state_dim]
            actions = torch.tanh(states @ action_projection)
            actions = torch.clamp(
                actions + 0.08 * torch.randn(self.horizon, self.action_dim, generator=self.generator),
                -1.0,
                1.0,
            )
            returns = prices[1:, : self.action_dim] - prices[:-1, : self.action_dim]
            turnover = torch.cat((actions[:1].abs(), (actions[1:] - actions[:-1]).abs()), dim=0)
            rewards = (actions * returns).sum(dim=-1) - 0.02 * turnover.sum(dim=-1)
            dones = torch.zeros(self.horizon)
            dones[-1] = 1.0
            episodes.append(EpisodeBatch(episode_id, states, actions, rewards, next_states, dones))
        return episodes

    @staticmethod
    def _columns(row: dict[str, str], prefixes: Sequence[str], fallback_count: int) -> list[str]:
        columns = [key for key in row if key.startswith(tuple(prefixes))]
        if columns:
            return sorted(columns)
        fallback = [f"{prefixes[0]}{index}" for index in range(fallback_count)]
        if fallback_count > 0 and all(column in row for column in fallback):
            return fallback
        if fallback_count == 0:
            return []
        raise ValueError(f"could not infer columns for prefixes {prefixes}")

    @staticmethod
    def _first_present(
        row: dict[str, str],
        candidates: Sequence[str],
        required: bool = True,
    ) -> str | None:
        for candidate in candidates:
            if candidate in row:
                return candidate
        if required:
            raise ValueError(f"missing required key from candidates {candidates}")
        return None

    @staticmethod
    def _tensor_from_columns(rows: Sequence[dict[str, str]], columns: Sequence[str]) -> Tensor:
        return torch.tensor(
            [[float(row[column]) for column in columns] for row in rows],
            dtype=torch.float32,
        )

    def _rewards_from_rows(self, rows: Sequence[dict[str, str]], actions: Tensor) -> Tensor:
        if "reward" in rows[0]:
            return torch.tensor([float(row["reward"]) for row in rows], dtype=torch.float32)
        if self.domain == "financial" and "return" in rows[0]:
            returns = torch.tensor([float(row["return"]) for row in rows], dtype=torch.float32)
            return returns * actions.mean(dim=-1)
        return -0.05 * actions.square().sum(dim=-1)


@dataclass(frozen=True)
class ShardAssignment:
    shard_id: int
    episode_ids: tuple[int, ...]


class SISAShardIndex:
    """Episode-consistent shard index for SISA retraining workflows."""

    def __init__(self, dataset: TrajectoryDataset, shard_count: int) -> None:
        if shard_count <= 0:
            raise ValueError("shard_count must be positive")
        if dataset.shard_count != shard_count:
            dataset = TrajectoryDataset([dataset[index] for index in range(len(dataset))], shard_count)
        self.dataset = dataset
        self.shard_count = shard_count
        self._assignments = self._build_assignments()

    def _build_assignments(self) -> dict[int, list[int]]:
        return {
            shard_id: list(self.dataset.shard_episode_ids(shard_id))
            for shard_id in self.dataset.shard_ids()
        }

    def shard_for_episode(self, episode_id: int) -> int:
        return episode_id % self.shard_count

    def affected_shards(self, deleted_episode_ids: Sequence[int]) -> set[int]:
        return {self.shard_for_episode(episode_id) for episode_id in deleted_episode_ids}

    def assignments(self) -> list[ShardAssignment]:
        return [
            ShardAssignment(shard_id=shard_id, episode_ids=tuple(episode_ids))
            for shard_id, episode_ids in sorted(self._assignments.items())
        ]

    def shard_dataset(self, shard_id: int) -> TrajectoryDataset:
        if shard_id not in self._assignments:
            raise KeyError(f"unknown shard_id {shard_id}")
        return TrajectoryDataset(
            [self.dataset.get_episode(episode_id) for episode_id in self._assignments[shard_id]]
        )
