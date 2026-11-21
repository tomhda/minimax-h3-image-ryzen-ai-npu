"""Weight layouts and CPU entry/exit operations for MiniMax-H3-Image."""
import numpy as np
import torch
import torch.nn.functional as F

from common import IMAGE_ROWS, PROJECTIONS

WIDTH = 5376
HEADS = 56
HEAD_DIM = 128
FFN_WIDTH = 14336
TEXT_LIMIT = 256
FRAME_ROWS = IMAGE_ROWS + TEXT_LIMIT
PROJECTION_KEYS = dict(zip(PROJECTIONS, ("attn.qkv_proj.weight", "attn.out_proj.weight",
                                        "mlp.fc1.weight", "mlp.fc2.weight")))
SMALL_KEYS = ("norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight")
ROPE_ORDER = np.concatenate((np.stack((np.arange(48), np.arange(48, 96)), axis=-1).reshape(-1),
                             np.arange(96, HEAD_DIM)))


class Outer:
    def __init__(self, parts, text_path, item, checkpoint=None):
        with np.load(parts.outer / "modules.npz") as arrays:
            self.weights = {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}
        self.text = torch.from_numpy(np.load(text_path / "refined.npy"))
        self.nt = self.text.shape[1]
        self.rows = self.nt + item["image_rows"]
        self.height, self.width = item["latent_height"], item["latent_width"]
        self.tembs = torch.from_numpy(np.load(parts.outer / "tembs.npy"))
        self.sigmas = torch.from_numpy(np.load(parts.outer / "sigmas.npy"))
        if checkpoint:
            self.latents = torch.from_numpy(np.load(checkpoint))
        else:
            noise = torch.randn((1, 24, 1, self.height, self.width),
                                generator=torch.Generator(device="cpu").manual_seed(item["seed"]), dtype=torch.float32)
            self.latents = noise.reshape(1, 24, 1, self.height // 2, 2, self.width // 2, 2)
            self.latents = self.latents.permute(0, 2, 3, 5, 1, 4, 6)
            self.latents = self.latents.reshape(-1, 96).contiguous()
        with np.load(text_path / "geometry.npz") as geometry:
            self.geometry = {key: torch.from_numpy(geometry[key]) for key in geometry.files}
        self.parts = parts

    def begin(self, step):
        weights = self.weights
        images = F.linear(self.latents[None].float(), weights["proj.weight"], weights["proj.bias"])
        hidden = self.text.new_zeros((1, self.rows, WIDTH))
        hidden = hidden.index_copy(1, torch.arange(self.nt), self.text)
        hidden = hidden.index_copy(1, torch.arange(self.nt, self.rows), images)
        return hidden[0], torch.from_numpy(np.load(self.parts.outer / f"modulations{step}.npy"))

    def end(self, step, tensor):
        weights = self.weights
        shift, scale = F.linear(F.silu(self.tembs[step]), weights["norm.linear.weight"],
                                weights["norm.linear.bias"]).chunk(2, -1)
        indices = torch.zeros(self.rows, dtype=torch.long)
        normalized = F.rms_norm(tensor[None], (WIDTH,), weights["norm.norm.weight"], 1e-5)
        normalized = normalized * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)
        velocity = F.linear(normalized, weights[f"head{step}.weight"], weights[f"head{step}.bias"])
        velocity = velocity.index_select(1, torch.arange(self.nt, self.rows))[0]
        self.latents = self.latents.float() + (self.sigmas[step] - self.sigmas[step + 1]) * velocity.float()

    def vae_input(self):
        tiles = self.latents.reshape(-1, self.height // 2, self.width // 2, 24, 2, 2)
        return tiles.permute(0, 3, 1, 4, 2, 5).reshape(-1, 24, self.height, self.width).contiguous()

