"""Regression test for InfiniteDataReader in training mode.

`_iter_one_dataset` restarts itself with `yield from` to make the dataset
infinite. That grows the generator delegation chain by one frame per pass
over the episode list and never unwinds, so a long training run eventually
dies with RecursionError.

This test uses a synthetic handler with a single one-sample episode, so one
pass over the episode list costs one sample. Pulling more samples than the
interpreter's recursion limit therefore crosses the old crash point.
"""
import json
import sys

import pytest
import torch

from datasets.dataset import InfiniteDataReader
from datasets.domain_handler.base import DomainHandler

# One sample per pass, so samples-requested maps 1:1 onto delegation depth.
SAMPLES = sys.getrecursionlimit() * 2


class OneSamplePerPassHandler(DomainHandler):
    """Yields a single sample, with no dependence on real dataset files."""

    dataset_name = "regression-fixture"

    def iter_episode(self, traj_idx, *, num_actions, training, image_aug,
                     lang_aug_map, action_mode, **kwargs):
        yield {
            "language_instruction": "regression fixture",
            "image_input": torch.zeros(self.num_views, 3, 4, 4),
            "image_mask": torch.ones(self.num_views, dtype=torch.bool),
            "abs_trajectory": torch.zeros(num_actions + 1, 20),
        }


@pytest.fixture
def metas_dir(tmp_path):
    meta = {
        "dataset_name": "regression-fixture",
        "robot_type": OneSamplePerPassHandler.dataset_name,
        "datalist": ["traj0.h5"],
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return str(tmp_path)


def test_training_mode_restarts_without_growing_the_stack(monkeypatch, metas_dir):
    monkeypatch.setattr(
        "datasets.dataset.get_handler_cls", lambda name: OneSamplePerPassHandler
    )

    reader = InfiniteDataReader(
        metas_path=metas_dir, num_actions=4, num_views=1, training=True
    )

    for i, sample in enumerate(reader):
        assert sample["proprio"].shape == (20,)
        assert sample["action"].shape == (4, 20)
        if i + 1 == SAMPLES:
            break
    else:  # pragma: no cover - training mode must never end
        pytest.fail("InfiniteDataReader stopped yielding in training mode")
