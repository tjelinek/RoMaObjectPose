from romatch.benchmarks.base_benchmark import Benchmark
from romatch.datasets import MegadepthBuilder
from torch.utils.data import ConcatDataset


class MegadepthDenseBenchmark(Benchmark):

    def __init__(self, data_root="data/megadepth", h=384, w=512, num_samples=2000) -> None:
        super().__init__('megadepth')
        mega = MegadepthBuilder(data_root=data_root)
        self.dataset = ConcatDataset(
            mega.build_scenes(split="test_loftr", ht=h, wt=w)
        )  # fixed resolution of 384,512
        self.num_samples = num_samples

