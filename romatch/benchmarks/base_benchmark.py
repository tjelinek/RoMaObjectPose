import torch
import tqdm
from torchvision import transforms

import romatch
from romatch.utils import warp_kpts


def geometric_dist(depth1, depth2, T_1to2, K1, K2, dense_matches):
    b, h1, w1, d = dense_matches.shape
    with torch.no_grad():
        x1 = dense_matches[..., :2].reshape(b, h1 * w1, 2)
        mask, x2 = warp_kpts(
            x1.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
        )
        x2 = torch.stack(
            (w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1
        )
        prob = mask.float().reshape(b, h1, w1)
    x2_hat = dense_matches[..., 2:]
    x2_hat = torch.stack(
        (w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
    )
    gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
    gd = gd[prob == 1]
    pck_1 = (gd < 1.0).float().mean()
    pck_3 = (gd < 3.0).float().mean()
    pck_5 = (gd < 5.0).float().mean()
    return gd, pck_1, pck_3, pck_5, prob


class Benchmark:

    def __init__(self, benchmark_prefix: str) -> None:
        self.benchmark_prefix = benchmark_prefix

    def benchmark(self, model, batch_size=8):
        model.train(False)
        with torch.no_grad():
            gd_tot = 0.0
            pck_1_tot = 0.0
            pck_3_tot = 0.0
            pck_5_tot = 0.0
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.ones(len(self.dataset)), replacement=False, num_samples=self.num_samples
            )
            B = batch_size
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=B, num_workers=batch_size, sampler=sampler
            )
            for idx, data in tqdm.tqdm(enumerate(dataloader), disable = romatch.RANK > 0):
                im_A, im_B, depth1, depth2, T_1to2, K1, K2 = (
                    data["im_A"].cuda(),
                    data["im_B"].cuda(),
                    data["im_A_depth"].cuda(),
                    data["im_B_depth"].cuda(),
                    data["T_1to2"].cuda(),
                    data["K1"].cuda(),
                    data["K2"].cuda(),
                )

                to_pil = transforms.ToPILImage()

                # Convert batched tensors to list of PIL images
                pil_A_list = [to_pil(im_A[i].detach().cpu()) for i in range(im_A.size(0))]
                pil_B_list = [to_pil(im_B[i].detach().cpu()) for i in range(im_B.size(0))]

                # Process each pair individually with batched=False
                matches_list = []
                certainty_list = []

                for pil_A, pil_B in zip(pil_A_list, pil_B_list):
                    match, cert = model.match(pil_A, pil_B, batched=False)
                    matches_list.append(match)
                    certainty_list.append(cert)

                # If you need to combine results back into batched format
                matches = torch.stack(matches_list) if matches_list[0] is not None else None
                certainty = torch.stack(certainty_list) if certainty_list[0] is not None else None

                # matches, certainty = model.match(im_A, im_B, batched=True)
                gd, pck_1, pck_3, pck_5, prob = geometric_dist(depth1, depth2, T_1to2, K1, K2, matches)

                if romatch.DEBUG_MODE:
                    from romatch.utils.utils import tensor_to_pil
                    import torch.nn.functional as F
                    path = "vis"
                    H, W = model.get_output_resolution()
                    white_im = torch.ones((B,1,H,W),device="cuda")
                    im_B_transfer_rgb = F.grid_sample(
                        im_B.cuda(), matches[:,:,:W, 2:], mode="bilinear", align_corners=False
                    )
                    warp_im = im_B_transfer_rgb
                    c_b = certainty[:,None]
                    vis_im = c_b * warp_im + (1 - c_b) * white_im
                    for b in range(B):
                        import os
                        os.makedirs(f"{path}/{model.name}/{idx}_{b}_{H}_{W}",exist_ok=True)
                        tensor_to_pil(vis_im[b], unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/warp.jpg")
                        tensor_to_pil(im_A[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_A.jpg")
                        tensor_to_pil(im_B[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_B.jpg")


                gd_tot, pck_1_tot, pck_3_tot, pck_5_tot = (
                    gd_tot + gd.mean(),
                    pck_1_tot + pck_1,
                    pck_3_tot + pck_3,
                    pck_5_tot + pck_5,
                )
        return {
            f"{self.benchmark_prefix}_epe": gd_tot.item() / len(dataloader),
            f"{self.benchmark_prefix}_pck_1": pck_1_tot.item() / len(dataloader),
            f"{self.benchmark_prefix}_pck_3": pck_3_tot.item() / len(dataloader),
            f"{self.benchmark_prefix}_pck_5": pck_5_tot.item() / len(dataloader),
        }
