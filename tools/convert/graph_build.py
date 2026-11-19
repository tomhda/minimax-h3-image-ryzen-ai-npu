"""Build the ONNX graphs for packed projections and DynamicDispatch attention."""
from common import PROJECTION_SHAPES

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

TILE = 256
GROUP = 128

def pack(values, scale, zero):
    from ryzenai_dynamic_dispatch import Attributes, matmulnbits

    n, k = values.shape
    raw = ((values[:, ::2] & 15) | ((values[:, 1::2] & 15) << 4)).reshape(n, k // GROUP, GROUP // 2)
    if zero.shape[1] % 2:
        zero = np.pad(zero, ((0, 0), (0, 1)), constant_values=8)
    zeros = (zero[:, ::2] & 15) | ((zero[:, 1::2] & 15) << 4)
    attrs = Attributes()
    for name, value in {
        "K": k, "N": n, "lora": False, "allow_weights_pad": True, "mladf_version": "v2",
        "asymmetric_quant": True, "bias_en": True, "block_size": GROUP, "enable_ctrl_pkt": False,
    }.items():
        attrs.set(name, value)
    data, _, real_k, real_n = matmulnbits.matmulnbits_pack_const_float32(
        raw, np.zeros(n, np.float32), scale.reshape(-1), zeros.reshape(-1), attrs,
    )
    return data, int(real_k), int(real_n)


def packed_node(prefix, source, target, values, scale, zero):
    n, k = values.shape
    data, real_k, real_n = pack(values, scale, zero)
    initializers = [nh.from_array(np.empty(0, dtype), prefix + name) for name, dtype in
                    (("W", np.uint8), ("S", np.float32), ("Z", np.uint8), ("B", np.float32))]
    initializers.append(h.make_tensor(prefix + "P", T.INT8, data.shape, data.tobytes(), raw=True))
    node = h.make_node("MatMulNBitsBf", [source, *(tensor.name for tensor in initializers)], [target],
                       domain="com.ryzenai", name=prefix + "npu", bits=4, block_size=GROUP,
                       K=k, N=n, real_K=real_k, real_N=real_n, mladf_version="v2")
    for name in ("hybrid_llm_cast_input", "hybrid_llm_cast_output"):
        node.attribute.append(h.make_attribute(name, [], attr_type=onnx.AttributeProto.INTS))
    return node, initializers


def make_int4(name, nodes, inputs, outputs, initializers, values):
    model = h.make_model(h.make_graph(nodes, name, inputs, outputs, initializers, value_info=values),
                         opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.ryzenai", 1)])
    model.ir_version = 10
    return model


def save_packed(model, directory):
    from ryzenai_onnx_utils.matcher import get_extractor
    from ryzenai_onnx_utils.passes.llm.jit import save_jit_weights

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "replaced.onnx"
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="replaced.onnx.data", size_threshold=1024)
    save_jit_weights(get_extractor(model), path, False, False, False, False)


def dit_projection(name, weights):
    k, n, specs = PROJECTION_SHAPES[name]
    values, scales, zeros = weights
    if values.shape != (n, k):
        raise ValueError(f"Unexpected {name} weight shape: {values.shape}")
    cap = max(k0 + block_k for k0, _, block_k, _ in specs)
    nodes, init, values_info, slices, outputs, pieces = [], [], [], [], [], []
    for index, (k0, n0, block_k, block_n) in enumerate(specs):
        ac, ar = min(block_k, k - k0), min(block_n, n - n0)
        q = np.full((block_n, block_k), 8, np.uint8)
        scale = np.full((block_n, block_k // GROUP), 1 / 32, np.float32)
        zero = np.full((block_n, block_k // GROUP), 8, np.uint8)
        q[:ar, :ac] = values[n0:n0 + ar, k0:k0 + ac]
        scale[:ar, :ac // GROUP] = scales[n0:n0 + ar, k0 // GROUP:(k0 + ac) // GROUP]
        zero[:ar, :ac // GROUP] = zeros[n0:n0 + ar, k0 // GROUP:(k0 + ac) // GROUP]
        p = f"b{index}_"
        for token, value in (("starts", [k0]), ("ends", [k0 + block_k]), ("axes", [2]), ("steps", [1])):
            init.append(nh.from_array(np.array(value, np.int64), p + token))
        nodes.append(h.make_node("Slice", ["A", *(p + s for s in ("starts", "ends", "axes", "steps"))],
                                 [p + "slice"], name=p + "slice_node"))
        if ac < block_k:
            init.append(nh.from_array(np.array([0, 0, 0, 0, 0, block_k - ac], np.int64), p + "pads"))
            nodes.append(h.make_node("Identity", [p + "slice"], [p + "input"], name=p + "pad_node"))
        else:
            nodes.append(h.make_node("Identity", [p + "slice"], [p + "input"]))
        nodes.append(h.make_node("Identity", [p + "input"], [p + "A"], name=p + "cast_in"))
        node, packed = packed_node(p, p + "A", p + "Y", q, scale, zero)
        init.extend(packed)
        if ar < block_n:
            for token, value in (("ns", [0]), ("ne", [ar]), ("na", [2])):
                init.append(nh.from_array(np.array(value, np.int64), p + token))
        nodes.append(node)
        for suffix in ("input", "A"):
            values_info.append(h.make_tensor_value_info(p + suffix, T.BFLOAT16, [1, TILE, block_k]))
        slices.append(h.make_tensor_value_info(p + "slice", T.BFLOAT16, [1, TILE, block_k]))
        outputs.append(h.make_tensor_value_info(p + "Y", T.BFLOAT16, [1, TILE, block_n]))
        pieces.append({"output": p + "Y", "N": block_n, "n0": n0})
    inputs = [h.make_tensor_value_info("A", T.BFLOAT16, [1, TILE, cap])]
    return make_int4("int4_projection", nodes, inputs, outputs, init, values_info + slices), {
        "K": k, "N": n, "cap": cap, "pieces": pieces,
    }


def vae_projection(weight):
    n, k = weight.shape
    groups = weight.reshape(n, -1, GROUP)
    lo, hi = groups.min(2), groups.max(2)
    center = (lo + hi) / 2
    best = None
    for alpha in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5):
        half = (hi - lo) / 2 * alpha
        lower = center - half
        scale = 2 * half / 15
        scale = np.where(scale == 0, 1, scale)
        zero = np.rint(-lower / scale).clip(0, 15).astype(np.uint8)
        values = np.rint(groups / scale[:, :, None] + zero[:, :, None]).clip(0, 15).astype(np.uint8)
        restored = (values.astype(np.float32) - zero[:, :, None]) * scale[:, :, None]
        error = np.square(restored - groups).sum(2)
        if best is None:
            best, best_error = (values, scale, zero), error
        else:
            improved = error < best_error
            best = (np.where(improved[:, :, None], values, best[0]), np.where(improved, scale, best[1]),
                    np.where(improved, zero, best[2]))
            best_error = np.minimum(best_error, error)
    q, scale, zero = best
    node, init = packed_node("", "Ab", "Yb", q.reshape(n, k), scale.astype(np.float32), zero)
    nodes = [h.make_node("Cast", ["A"], ["Ab"], name="cast_in", to=T.BFLOAT16),
             node, h.make_node("Cast", ["Yb"], ["Y"], name="cast_out", to=T.FLOAT)]
    return make_int4("vae_linear", nodes, [h.make_tensor_value_info("A", T.FLOAT, [1, TILE, k])],
                     [h.make_tensor_value_info("Y", T.FLOAT, [1, TILE, n])], init,
                     [h.make_tensor_value_info("Ab", T.BFLOAT16, [1, TILE, k]),
                      h.make_tensor_value_info("Yb", T.BFLOAT16, [1, TILE, n])])


class AttentionGraph:
    def __init__(self, shape):
        self.shape = shape
        self.nodes, self.init, self.info, self.inputs = [], [], [], []

    def input(self, name, shape=None):
        self.inputs.append(h.make_tensor_value_info(name, T.FLOAT, shape or self.shape))

    def const(self, name, value):
        self.init.append(nh.from_array(value, name))

    def op(self, kind, inputs, name, shape=None, **attributes):
        self.nodes.append(h.make_node(kind, inputs, [name], name=name, **attributes))
        self.info.append(h.make_tensor_value_info(name, T.FLOAT, shape or self.shape))

    def model(self, name):
        graph = h.make_graph(self.nodes, name, self.inputs, [h.make_tensor_value_info("Y", T.FLOAT, self.shape)],
                             self.init, value_info=self.info)
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18), h.make_opsetid("com.microsoft", 1)])
        model.ir_version = 9
        return model


def dit_attention(q_scale, k_scale, order):
    g = AttentionGraph([1, 1280, 3072])
    for name in ("q", "k", "v", "mark"):
        g.input(name)
    for name in ("cos", "sin"):
        g.input(name + "_image", [1024, 64])
        g.input(name + "_text", [256, 64])
    g.const("positions", np.arange(1280, dtype=np.int64)[None])
    g.const("eye", np.eye(3072, dtype=np.float32))
    g.const("rn_flat", np.array([1, 1280, 3072], np.int64))
    g.const("rn_heads", np.array([1, 1280, 24, 128], np.int64))
    # Preserve value-info order used by the Ryzen AI partition passes.
    norm_info = []
    for name, weight in (("q", q_scale), ("k", k_scale)):
        g.const(name + "_norm_weight", weight[order].astype(np.float32))
        g.op("MatMul", [name, "eye"], name + "_rn_mm")
        g.op("Reshape", [name + "_rn_mm", "rn_heads"], name + "_rn_heads", [1, 1280, 24, 128])
        g.op("SimplifiedLayerNormalization", [name + "_rn_heads", name + "_norm_weight"],
             name + "_rn_normalized", [1, 1280, 24, 128], axis=-1, epsilon=1e-5)
        g.op("Reshape", [name + "_rn_normalized", "rn_flat"], name + "_rn_flat")
    norm_info, g.info = g.info, []
    for name in ("cos", "sin"):
        g.op("Concat", [name + "_image", name + "_text"], name, [1280, 64], axis=0)
    g.op("MatMul", ["v", "eye"], "v_real")
    g.op("MatMul", ["mark", "eye"], "v_mark")
    for tag, value in (("y0", "v_real"), ("a", "v_mark")):
        for name in ("q", "k"):
            g.op("RotaryEmbedding", [name + "_rn_flat", "positions", "cos", "sin"], tag + "_" + name,
                 domain="com.microsoft", num_heads=24, interleaved=1, rotary_embedding_dim=128)
        g.op("MultiHeadAttention", [tag + "_q", tag + "_k", value], tag, domain="com.microsoft", num_heads=24)
    g.op("Div", ["y0", "a"], "Y")
    g.info.extend(norm_info)
    return g.model("heads24_m1280")


def vae_attention():
    g = AttentionGraph([1, 1024, 1536])
    for name in ("q", "k", "v", "mark"):
        g.input(name)
    g.const("eye", np.eye(1536, dtype=np.float32))
    for tag, value in (("real", "v"), ("mass", "mark")):
        projected = []
        for name in ("q", "k", value):
            output = tag + "_" + name
            g.op("MatMul", [name, "eye"], output)
            g.nodes[-1].name = output + "_identity"
            projected.append(output)
        g.op("MultiHeadAttention", projected, tag, domain="com.microsoft", num_heads=24)
    g.nodes.append(h.make_node("Div", ["real", "mass"], ["Y"], name="padding_correction"))
    return g.model("vae_attention")


def save_casts(directory):
    for name, source, target in (("to_bf", T.FLOAT, T.BFLOAT16), ("to_fp", T.BFLOAT16, T.FLOAT)):
        node = h.make_node("CastAvx", ["A"], ["Y"], domain="com.ryzenai", to=target)
        shape = [1, "M", "K"]
        model = make_int4("host_cast", [node], [h.make_tensor_value_info("A", source, shape)],
                          [h.make_tensor_value_info("Y", target, shape)], [], [])
        onnx.save_model(model, directory / f"{name}.onnx")

