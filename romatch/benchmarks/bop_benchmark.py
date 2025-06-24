from typing import List
from pathlib import Path


from romatch.benchmarks.base_benchmark import Benchmark
from romatch.datasets.bop import BOPBuilder
from torch.utils.data import ConcatDataset


class BOPBenchmark(Benchmark):

    def __init__(self, data_root: Path, datasets: List, h=384, w=512, num_samples=2000) -> None:
        super().__init__('bop')

        bop = BOPBuilder(data_root=data_root)

        self.dataset = ConcatDataset(
            bop.build_scenes(dataset=dataset, split="test", ht=h, wt=w) for dataset in datasets
        )  # fixed resolution of 384,512
        self.num_samples = num_samples

