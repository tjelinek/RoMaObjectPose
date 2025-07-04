import os
from pathlib import Path

import cv2
import numpy as np
import torch

from romatch.datasets.scene import SceneBuilder, Scene


class BopScene(Scene):
    def __init__(self, data_root, scene_info, ht=384, wt=512, min_overlap=0.0, max_overlap=1.0, shake_t=0,
                 normalize=True, max_num_pairs=100_000, scene_name=None, use_horizontal_flip_aug=False,
                 use_single_horizontal_flip_aug=False, colorjiggle_params=None, random_eraser=None, use_randaug=False,
                 randomize_size=False) -> None:
        super().__init__(data_root, scene_info, ht, wt, min_overlap, max_overlap, shake_t, normalize, max_num_pairs,
                         scene_name, use_horizontal_flip_aug, use_single_horizontal_flip_aug, colorjiggle_params,
                         random_eraser, use_randaug, randomize_size)
        self.depth_scale_to_meter = scene_info['depth_scale_to_meter']

    def load_depth(self, depth_ref, crop=None):
        depth = cv2.imread(str(depth_ref), cv2.IMREAD_UNCHANGED)  # uint16
        depth_t = torch.from_numpy(depth).to(torch.float32) * self.depth_scale_to_meter
        return depth_t.clone()


class BOPBuilder(SceneBuilder):
    def __init__(self, data_root=Path("/mnt/personal/jelint19/data/bop")) -> None:
        super().__init__(data_root)
        self.data_root: Path = data_root

    def build_scenes(self, dataset: str, split="train", min_overlap=0.0, scene_names=None, **kwargs):
        path_to_scenes = self.data_root / dataset / 'train_pbr'
        all_scenes = np.array(sorted(os.listdir(path_to_scenes)))

        np.random.seed(42)
        indices = np.arange(len(all_scenes))
        np.random.shuffle(indices)
        train_delim = int(len(all_scenes) * 0.7)

        train_split = indices[:train_delim]
        val_split = indices[train_delim:]

        train_scenes = all_scenes[train_split]
        val_scenes = all_scenes[val_split]

        if split == "train":
            scene_names = train_scenes
        elif split == "test":
            scene_names = val_scenes
        elif split == "custom":
            scene_names = scene_names
        else:
            raise ValueError(f"Split {split} not available")

        scenes = []
        for scene_name in sorted(scene_names):
            scene_info_path = path_to_scenes / scene_name / 'scene_info.npy'

            scene_info = np.load(scene_info_path, allow_pickle=True).item()

            scenes.append(
                BopScene(self.data_root, scene_info, min_overlap=min_overlap, scene_name=scene_name, **kwargs))
        return scenes
