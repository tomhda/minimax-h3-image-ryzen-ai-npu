# Refiner weight mapping adapted and modified from Diffusers scripts/convert_minimax_h3_to_diffusers.py.
"""Stream Qwen3-VL text layers and prepare the transformer inputs."""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

START = time.perf_counter()

import numpy as np
import torch
from transformers import Qwen2TokenizerFast, Qwen3VLConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding

from common import Parts, add_times, clock, duration, load_config, read_json, save_json
from mmap_weights import Weights
from model import FFN_WIDTH, HEAD_DIM, HEADS, TEXT_LIMIT, WIDTH

TEXT_LAYERS = 50
EMBED = "model.language_model.embed_tokens.weight"
IMPORT_S = time.perf_counter() - START


def load_text_layer(reader, index, keys, read_workers):
    start = clock()
    base = f"model.language_model.layers.{index}."

    def read(key):
        return key.removeprefix(base), reader.read(key, direct=True)

    with ThreadPoolExecutor(max_workers=read_workers) as workers:
        state = dict(workers.map(read, keys))
    return state, duration(start)


def encode(encoder, tokens, read_workers, record):
    start = clock()
    reader = Weights(encoder)
    layer_keys = [[key for key in reader.mapping if key.startswith(f"model.language_model.layers.{i}.")]
                  for i in range(TEXT_LAYERS)]
    reader.preload_headers([EMBED, *(key for keys in layer_keys for key in keys)])
    config = Qwen3VLConfig.from_pretrained(str(encoder), local_files_only=True).text_config
    config._attn_implementation = "eager"
    hidden = reader.read(EMBED, tokens).float().unsqueeze(0).clone()
    record["embedding_and_index"] = duration(start)

    rotary = Qwen3VLTextRotaryEmbedding(config)
    positions = torch.arange(len(tokens)).view(1, 1, -1).expand(4, 1, -1).clone()
    text_positions = positions[0]
    pos_embeddings = rotary(hidden, positions[1:])
    mask = create_causal_mask(config=config, inputs_embeds=hidden,
                              attention_mask=torch.ones(1, len(tokens), dtype=torch.long),
                              past_key_values=None, position_ids=text_positions)

    rows = []
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        pending = prefetch.submit(load_text_layer, reader, 0, layer_keys[0], read_workers)
        for index in range(TEXT_LAYERS):
            waiting = clock()
            state, reading = pending.result()
            wait = duration(waiting)
            start = clock()
            for key in state:
                state[key] = state[key].float()
            with torch.device("meta"):
                layer = Qwen3VLTextDecoderLayer(config, index)
            layer.load_state_dict(state, strict=True, assign=True)
            layer.eval()
            del state
            conversion = duration(start)
            if index + 1 < TEXT_LAYERS:
                pending = prefetch.submit(load_text_layer, reader, index + 1, layer_keys[index + 1], read_workers)
            start = clock()
            hidden = layer(hidden, attention_mask=mask, position_ids=text_positions, past_key_values=None,
                           position_embeddings=pos_embeddings, use_cache=False)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            rows.append({"layer": index, "read": reading, "weight_conversion": conversion,
                         "read_wait": wait, "inference": duration(start)})
            del layer
    record["layers"] = rows
    return hidden.float().numpy()


def refiner_weights(reader):
    state = {}
    for key in reader.mapping:
        if not key.startswith("token_refiner."):
            continue
        value = reader.read(key)
        target = key.removeprefix("token_refiner.").replace("blocks.", "refiner_blocks.", 1)
        target = target.replace(".attn.q_norm.", ".attn.norm_q.")
        target = target.replace(".attn.k_norm.", ".attn.norm_k.")
        target = target.replace(".attn.out_proj.", ".attn.to_out.0.")
        if target.endswith(".attn.qkv_proj.weight"):
            base = target.removesuffix("qkv_proj.weight")
            for name, projection in zip(("q", "k", "v"), value.reshape(HEADS, 3, HEAD_DIM, -1).unbind(1)):
                state[f"{base}to_{name}.weight"] = projection.reshape(HEADS * HEAD_DIM, -1).float().contiguous()
        elif target.endswith(".mlp.fc1.weight"):
            gate, value = value.chunk(2, dim=0)
            state[target.replace(".mlp.fc1.", ".ff.net.0.proj.")] = torch.cat((value, gate)).float().contiguous()
        else:
            state[target.replace(".mlp.fc2.", ".ff.net.2.")] = value.float().contiguous()
    return state


def refine(refiner_path, embeds, record):
    start = clock()
    from diffusers.models.transformers.transformer_minimax_h3_image import (
        MiniMaxH3ImageRotaryPosEmbed, MiniMaxH3ImageTokenRefiner,
    )
    from diffusers.modular_pipelines.minimax_h3_image.before_denoise import build_image_packed_sequence

    record["refiner_import"] = duration(start)
    start = clock()
    reader = Weights(refiner_path.parent, shard=refiner_path.name)
    with torch.device("meta"):
        context = torch.nn.Linear(5120, WIDTH)
        refiner = MiniMaxH3ImageTokenRefiner(
            hidden_size=WIDTH, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
            ffn_dim=FFN_WIDTH, num_layers=2, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
        )
    context.load_state_dict({name: reader.fp32(f"condition_proj.{name}") for name in ("weight", "bias")}, assign=True)
    refiner.load_state_dict(refiner_weights(reader), strict=True, assign=True)
    context.eval()
    refiner.eval()
    record["refiner_load"] = duration(start)

    start = clock()
    result = refiner(context(torch.from_numpy(embeds))).numpy()
    positions, tags, *_ = build_image_packed_sequence(torch.ones(result.shape[1], dtype=torch.long),
                                                     [], 64, 64, (1, 2, 2), 0)
    cos, sin = MiniMaxH3ImageRotaryPosEmbed(rope_freq_dim=16, rope_theta=10000.0)(positions)
    record["refiner_inference"] = duration(start)
    return result, dict(tags=tags.numpy(), cos=cos.numpy(), sin=sin.numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--item", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--read-workers", required=True, type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    record = {"import_s": IMPORT_S}
    model = config["model_path"]
    start = clock()
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(model / "tokenizer"), local_files_only=True)
    tokens = tokenizer(read_json(args.item)["prompt"], add_special_tokens=False)["input_ids"]
    record["tokenize"] = duration(start)
    record["tokens"] = len(tokens)
    if len(tokens) > TEXT_LIMIT:
        save_json(args.out / "text_log.json", record | {"status": "skipped"})
        return 3
    if not tokens:
        raise ValueError("Prompt contains no tokens")
    with torch.no_grad():
        embeds = encode(config["text_encoder_path"], tokens, args.read_workers, record)
        if not np.isfinite(embeds).all():
            raise RuntimeError("Non-finite text embeddings")
        refined, geometry = refine(Parts(config).refiner, embeds, record)
        if not np.isfinite(refined).all():
            raise RuntimeError("Non-finite refiner output")
    start = clock()
    np.save(args.out / "text_embeds.npy", embeds)
    np.save(args.out / "refined.npy", refined)
    np.savez(args.out / "geometry.npz", **geometry)
    record["save"] = duration(start)
    record["weights_load_and_convert"] = add_times(
        [record["embedding_and_index"], record["refiner_load"]]
        + [row[key] for row in record["layers"] for key in ("read", "weight_conversion")],
    )
    record["inference"] = add_times([record["refiner_inference"]] + [row["inference"] for row in record["layers"]])
    save_json(args.out / "text_log.json", record | {"status": "complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

