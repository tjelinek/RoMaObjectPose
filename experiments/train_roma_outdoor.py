import os
from pathlib import Path

import torch
from argparse import ArgumentParser

from torch import nn
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import wandb

from romatch.benchmarks import MegadepthDenseBenchmark
from romatch.datasets.bop import BOPBuilder
from romatch.datasets.megadepth import MegadepthBuilder
from romatch.losses.robust_loss import RobustLosses
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark, MegadepthDenseBenchmark, HpatchesHomogBenchmark

from romatch.train.train import train_k_steps
from romatch.models.matcher import *
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.models.encoders import *
from romatch.checkpointing import CheckPoint

resolutions = {"low": (448, 448), "medium": (14 * 8 * 5, 14 * 8 * 5), "high": (14 * 8 * 6, 14 * 8 * 6)}


def get_model(pretrained_backbone=True, resolution="medium", checkpoint_path=None, **kwargs):
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]),
        decoder_dim,
        cls_to_coord_res ** 2 + 1,
        is_classifier=True,
        amp=True,
        pos_enc=False, )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True

    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius=7,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "8": ConvRefiner(
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius=3,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "4": ConvRefiner(
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius=2,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "2": ConvRefiner(
                2 * 64 + 16,
                128 + 16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict({
        "16": proj16,
        "8": proj8,
        "4": proj4,
        "2": proj2,
        "1": proj1,
    })
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(coordinate_decoder,
                      gps,
                      proj,
                      conv_refiner,
                      detach=True,
                      scales=["16", "8", "4", "2", "1"],
                      displacement_dropout_p=displacement_dropout_p,
                      gm_warp_dropout_p=gm_warp_dropout_p)
    h, w = resolutions[resolution]
    encoder = CNNandDinov2(
        cnn_kwargs=dict(
            pretrained=pretrained_backbone,
            amp=True),
        amp=True,
        use_vgg=True,
    )
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, **kwargs)

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Remove 'module.' prefix if present (from DataParallel training)
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('module.'):
                breakpoint()
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = value

        # Load the state dict, allowing for some mismatched keys
        try:
            matcher.load_state_dict(new_state_dict, strict=True)
            print("Checkpoint loaded successfully (strict mode)")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}")
            print("Trying non-strict loading...")
            missing_keys, unexpected_keys = matcher.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")
                # print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                raise ValueError(f"Unexpected keys: {unexpected_keys}")
                # print(f"Unexpected keys: {unexpected_keys}")
            print("Checkpoint loaded with some mismatched keys")

    return matcher


def freeze_all_except_certainty(model):
    """
    Freeze all parameters except those that contribute to certainty prediction.

    Based on the forward pass, certainties come from:
    1. embedding_decoder (produces gm_certainty)
    2. conv_refiner (produces delta_certainty)
    """

    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze certainty-related components
    print("Unfreezing certainty-related components:")

    # 1. Unfreeze embedding_decoder (produces gm_certainty)
    if hasattr(model, 'embedding_decoder'):
        for param in model.embedding_decoder.parameters():
            param.requires_grad = True
        print("  ✓ embedding_decoder (produces gm_certainty)")

    # 2. Unfreeze conv_refiner (produces delta_certainty)
    if hasattr(model, 'conv_refiner'):
        for param in model.conv_refiner.parameters():
            param.requires_grad = True
        print("  ✓ conv_refiner (produces delta_certainty)")

    # 3. If the model has a decoder attribute that contains these components
    if hasattr(model, 'decoder'):
        if hasattr(model.decoder, 'embedding_decoder'):
            for param in model.decoder.embedding_decoder.parameters():
                param.requires_grad = True
            print("  ✓ decoder.embedding_decoder")

        if hasattr(model.decoder, 'conv_refiner'):
            for param in model.decoder.conv_refiner.parameters():
                param.requires_grad = True
            print("  ✓ decoder.conv_refiner")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params / total_params:.2%}")


def get_model_for_finetuning(checkpoint_path: Path, resolution="medium", freeze_backbone=False, **kwargs):
    """
    Load model from checkpoint specifically for fine-tuning.

    Args:
        checkpoint_path: Path to the checkpoint file
        resolution: Model resolution
        freeze_backbone: Whether to freeze the backbone encoder for fine-tuning
        **kwargs: Additional arguments
    """
    # Load the model architecture (without pretrained backbone since we're loading from checkpoint)
    matcher = get_model(pretrained_backbone=False, resolution=resolution,
                        checkpoint_path=checkpoint_path, **kwargs)

    # Optionally freeze parts of the model for fine-tuning
    freeze_all_except_certainty(matcher)
    if freeze_backbone:
        print("Freezing backbone encoder...")
        for param in matcher.encoder.parameters():
            param.requires_grad = False

    # You can also freeze specific components
    # For example, freeze only the CNN part but not DINOv2:
    # for param in matcher.encoder.cnn.parameters():
    #     param.requires_grad = False

    return matcher


def train(args):
    dist.init_process_group('nccl')
    #torch._dynamo.config.verbose=True
    gpus = int(os.environ['WORLD_SIZE'])
    # create model and move it to GPU with id rank
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}")
    device_id = rank % torch.cuda.device_count()
    romatch.LOCAL_RANK = device_id
    torch.cuda.set_device(device_id)
    
    resolution = args.train_resolution
    wandb_log = not args.dont_log_wandb
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    wandb_mode = "online" if wandb_log else "disabled"
    wandb.init(project="RoMa Certainty Fine-Tuning", entity='jelinek-vrg-fel-cvut-cz',
               name=experiment_name, reinit=False, mode=wandb_mode)
    roma_checkpoint_dir = Path("/mnt/personal/jelint19/weights/RoMa/")
    checkpoint_dir = roma_checkpoint_dir / "checkpoints/"
    pretrained_model_path = roma_checkpoint_dir / "roma_outdoor.pth"

    h, w = resolutions[resolution]
    # model = get_model(pretrained_backbone=True, resolution=resolution, attenuate_cert=False).to(dev)
    model = get_model_for_finetuning(pretrained_model_path)
    # Num steps
    global_step = 0
    batch_size = args.gpu_batch_size
    step_size = gpus*batch_size
    romatch.STEP_SIZE = step_size

    N = (32 * 250000)  # 250k steps of batch size 32
    # checkpoint every
    k = 25000 // romatch.STEP_SIZE

    # Data
    roma_data_root = Path("/mnt/personal/jelint19/data/roma_training")
    bop_data_root = Path("/mnt/personal/jelint19/data/bop")
    ho3d_data_root = Path("/mnt/personal/jelint19/data/HO3D")

    mega = MegadepthBuilder(data_root=roma_data_root, loftr_ignore=True, imc21_ignore=True)
    bop = BOPBuilder(data_root=bop_data_root)
    ho3d = HO3DBuilder(data_root=ho3d_data_root)

    use_horizontal_flip_aug = True
    rot_prob = 0
    depth_interpolation_mode = "bilinear"

    bop_train_handal = bop.build_scenes(dataset='handal', split='train', min_overlap=0.35, shake_t=32,
                                        use_horizontal_flip_aug=use_horizontal_flip_aug, ht=h, wt=w)
    bop_train_hope = bop.build_scenes(dataset='hope', split='train', min_overlap=0.35, shake_t=32,
                                      use_horizontal_flip_aug=use_horizontal_flip_aug, ht=h, wt=w)
    train_ho3d = ho3d.build_scenes(split="train", min_overlap=0.35, shake_t=32,
                                   use_horizontal_flip_aug=use_horizontal_flip_aug, ht=h, wt=w)

    fine_tuning_scenes = bop_train_handal + bop_train_hope + train_ho3d
    if args.train_also_on_megadepth:
        megadepth_train1 = mega.build_scenes(
            split="train_loftr", min_overlap=0.01, shake_t=32, use_horizontal_flip_aug=use_horizontal_flip_aug,
            rot_prob=rot_prob, ht=h, wt=w)
        megadepth_train2 = mega.build_scenes(
            split="train_loftr", min_overlap=0.35, shake_t=32, use_horizontal_flip_aug=use_horizontal_flip_aug,
            rot_prob=rot_prob, ht=h, wt=w)
        train_dataset = ConcatDataset(megadepth_train1 + megadepth_train2 + fine_tuning_scenes)
    else:
        train_dataset = ConcatDataset(fine_tuning_scenes)

    mega_ws = mega.weight_scenes(train_dataset, alpha=0.75)
    # Loss and optimizer
    depth_loss = RobustLosses(
        ce_weight=0.01,
        local_dist={1: 4, 2: 4, 4: 8, 8: 8},
        local_largest_scale=8,
        depth_interpolation_mode=depth_interpolation_mode,
        alpha=0.5,
        c=1e-4, )
    parameters = [
        {"params": model.encoder.parameters(), "lr": romatch.STEP_SIZE * 5e-6 / 8},
        {"params": model.decoder.parameters(), "lr": romatch.STEP_SIZE * 1e-4 / 8},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int((9 * N / romatch.STEP_SIZE) // 10)])
    megadense_benchmark = MegadepthDenseBenchmark(str(roma_data_root), num_samples=1000, h=h, w=w)
    checkpointer = CheckPoint(str(checkpoint_dir), experiment_name)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step)
    romatch.GLOBAL_STEP = global_step
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters = False, gradient_as_bucket_view=True)
    grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
    grad_clip_norm = 0.01
    for n in range(romatch.GLOBAL_STEP, N, k * romatch.STEP_SIZE):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples=batch_size * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=mega_sampler,
                num_workers=8,
            )
        )
        train_k_steps(
            n, k, mega_dataloader, ddp_model, depth_loss, optimizer, lr_scheduler, grad_scaler, grad_clip_norm = grad_clip_norm,
        )
        checkpointer.save(model, optimizer, lr_scheduler, romatch.GLOBAL_STEP)
        wandb.log(megadense_benchmark.benchmark(model), step=romatch.GLOBAL_STEP)


def test_mega_8_scenes(model, name):
    mega_8_scenes_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth",
                                                               scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                                            'mega_8_scenes_0025_0.1_0.3.npz',
                                                                            'mega_8_scenes_0021_0.1_0.3.npz',
                                                                            'mega_8_scenes_0008_0.1_0.3.npz',
                                                                            'mega_8_scenes_0032_0.1_0.3.npz',
                                                                            'mega_8_scenes_1589_0.1_0.3.npz',
                                                                            'mega_8_scenes_0063_0.1_0.3.npz',
                                                                            'mega_8_scenes_0024_0.1_0.3.npz',
                                                                            'mega_8_scenes_0019_0.3_0.5.npz',
                                                                            'mega_8_scenes_0025_0.3_0.5.npz',
                                                                            'mega_8_scenes_0021_0.3_0.5.npz',
                                                                            'mega_8_scenes_0008_0.3_0.5.npz',
                                                                            'mega_8_scenes_0032_0.3_0.5.npz',
                                                                            'mega_8_scenes_1589_0.3_0.5.npz',
                                                                            'mega_8_scenes_0063_0.3_0.5.npz',
                                                                            'mega_8_scenes_0024_0.3_0.5.npz'])
    mega_8_scenes_results = mega_8_scenes_benchmark.benchmark(model, model_name=name)
    print(mega_8_scenes_results)
    json.dump(mega_8_scenes_results, open(f"results/mega_8_scenes_{name}.json", "w"))


def test_mega1500(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_{name}.json", "w"))


def test_mega_dense(model, name):
    megadense_benchmark = MegadepthDenseBenchmark("data/megadepth", num_samples=1000)
    megadense_results = megadense_benchmark.benchmark(model)
    json.dump(megadense_results, open(f"results/mega_dense_{name}.json", "w"))


def test_hpatches(model, name):
    hpatches_benchmark = HpatchesHomogBenchmark("data/hpatches")
    hpatches_results = hpatches_benchmark.benchmark(model)
    json.dump(hpatches_results, open(f"results/hpatches_{name}.json", "w"))


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    import romatch

    parser = ArgumentParser()
    parser.add_argument("--only_test", action='store_true')
    parser.add_argument("--debug_mode", action='store_true', default=True)
    parser.add_argument("--dont_log_wandb", action='store_true')
    parser.add_argument("--train_resolution", default='medium')
    parser.add_argument("--gpu_batch_size", default=8, type=int)
    parser.add_argument("--wandb_entity", required=False)
    parser.add_argument("--train_also_on_megadepth", default=False, required=False)

    args, _ = parser.parse_known_args()
    romatch.DEBUG_MODE = args.debug_mode
    if not args.only_test:
        train(args)
