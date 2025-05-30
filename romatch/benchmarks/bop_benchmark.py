from typing import List

import torch
import tqdm
from torchvision import transforms

from romatch.benchmarks.base_benchmark import geometric_dist, Benchmark
from romatch.datasets.bop import BOPBuilder
from torch.utils.data import ConcatDataset
import romatch

class BOPBenchmark(Benchmark):

    def __init__(self, data_root, datasets: List, h=384, w=512, num_samples=2000) -> None:
        super().__init__()

        bop = BOPBuilder(data_root=data_root)

        self.dataset = ConcatDataset(
            bop.build_scenes(dataset=dataset, split="test", ht=h, wt=w) for dataset in datasets
        )  # fixed resolution of 384,512
        self.num_samples = num_samples

