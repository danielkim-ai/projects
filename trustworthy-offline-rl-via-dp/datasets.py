from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils import EpisodeBatch, make_synthetic_episodes


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
