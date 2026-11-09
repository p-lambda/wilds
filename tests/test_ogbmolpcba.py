"""Batch small molecular graphs without downloading the OGB archives."""
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader


@unittest.skipUnless(find_spec("torch_geometric"), "requires PyTorch Geometric")
class OGBBatchTests(unittest.TestCase):
    def setUp(self):
        from torch_geometric.data import Data

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for path in ("raw/scaffold_group.npy", "processed/group_assignment.npy"):
            (self.root / path).parent.mkdir()
            np.save(self.root / path, [7, 9])

        class TinyOGB(list):
            num_tasks = 2
            __num_classes__ = 2

            def get_idx_split(self):
                return {"train": [0, 1], "valid": [], "test": []}

        self.graphs = TinyOGB([Data(x=torch.tensor([[1], [2]])),
                               Data(x=torch.tensor([[3]]))])
        self.graphs.root = str(self.root)
        self.graphs.data = Data(y=torch.tensor([[0., 1.], [1., 0.]]))

    def load_batch(self, module, split, unlabeled=False):
        with patch(module + ".PygGraphPropPredDataset", return_value=self.graphs):
            dataset = get_dataset("ogb-molpcba", root_dir=str(self.root), unlabeled=unlabeled)
        subset = dataset.get_subset(split)
        return next(iter(get_eval_loader("standard", subset, batch_size=2)))

    def assert_graphs_and_scaffolds(self, graphs, metadata):
        self.assertEqual(graphs.x.tolist(), [[1], [2], [3]])
        self.assertEqual(graphs.batch.tolist(), [0, 0, 1])
        self.assertEqual(metadata[:, 0].tolist(), [7, 9])

    def test_labeled_batch(self):
        graphs, labels, metadata = self.load_batch("wilds.datasets.ogbmolpcba_dataset", "train")
        self.assert_graphs_and_scaffolds(graphs, metadata)
        self.assertEqual(labels.tolist(), [[0., 1.], [1., 0.]])

    def test_unlabeled_batch(self):
        graphs, metadata = self.load_batch("wilds.datasets.unlabeled.ogbmolpcba_unlabeled_dataset",
                                           "train_unlabeled", unlabeled=True)
        self.assert_graphs_and_scaffolds(graphs, metadata)
