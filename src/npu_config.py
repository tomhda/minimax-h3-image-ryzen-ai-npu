"""Ryzen AI Light EP setup for attention, int4 projections and host casts."""
import ctypes
import os
from pathlib import Path

import onnxruntime as ort

from common import read_json
from windows_runtime import native_directory

PROVIDER = "RyzenAILightExecutionProvider"


class Npu:
    def __init__(self, config, threads=2):
        self.deployment = native_directory(config["ryzen_ai_path"]) / "deployment"
        self.dd = native_directory(config["genai_sd_path"]) / "lib"
        roots = {config["parts_path"], *config.get("part_paths", {}).values()}
        self.part_roots = sorted((root for root in roots if not str(root).isascii() and root.is_dir()),
                                 key=lambda root: len(root.parts), reverse=True)
        self.dll = self.deployment / "onnxruntime_providers_ryzenai.dll"
        self.threads = threads
        self.sd_env()
        self.dll_directory = os.add_dll_directory(str(self.deployment))
        self.library = ctypes.CDLL(str(self.dll))
        previous = Path.cwd()
        try:
            os.chdir(self.deployment)
            ort.register_execution_provider_library(PROVIDER, str(self.dll))
        finally:
            os.chdir(previous)
        self.devices = [device for device in ort.get_ep_devices()
                        if device.ep_name == PROVIDER and device.device.type == ort.OrtHardwareDeviceType.NPU]
        if not self.devices:
            raise RuntimeError("The Ryzen AI Light EP did not find an NPU")

    def sd_env(self):
        os.environ.update(DD_ROOT=str(self.dd),
                          DD_PLUGINS_ROOT=str(self.dd / "transaction" / "stx"))

    def int_env(self):
        os.environ["DD_ROOT"] = str(self.deployment)
        os.environ.pop("DD_PLUGINS_ROOT", None)

    def model_path(self, path):
        path = Path(path).absolute()
        if str(path).isascii():
            return path
        for root in self.part_roots:
            if path.is_relative_to(root):
                return native_directory(root) / path.relative_to(root)
        return native_directory(path.parent) / path.name

    def session(self, path, kind, compile_part=False):
        path = self.model_path(path)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cache = {"dd_cache": str(path.parent / "cache")}
        model = str(path)
        if kind in ("attention", "cast"):
            self.sd_env()
            options.add_provider_for_devices(self.devices, {
                **cache, "compile_fusion_rt": str(int(compile_part)), "onnx_custom_ops_const_key": ""
            })
        elif kind == "dit":
            # Keep the SD plugin selection active, as required by the attention session.
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.add_provider_for_devices(self.devices, {})
            for key, value in {
                **cache, "external_data_file": str(path.with_suffix(".pb.bin")),
                "hybrid_opt_max_seq_length": "4096", "hybrid_opt_token_backend": "npu",
                "custom_allocator": "ryzen_mm",
            }.items():
                options.add_session_config_entry(key, value)
            refs = path.with_name("source_refs.json")
            if refs.is_file():
                # An existing graph can refer to constants stored beside its source graph.
                import onnx

                mapping = {name: Path(value).resolve() for name, value in read_json(refs).items()}
                base = Path(os.path.commonpath(list(mapping.values()))).parent
                graph = onnx.load(path, load_external_data=False)
                for initializer in graph.graph.initializer:
                    for entry in initializer.external_data:
                        if entry.key == "location":
                            entry.value = mapping[entry.value].relative_to(base).as_posix()
                options.add_session_config_entry("session.model_external_initializers_file_folder_path", str(native_directory(base)))
                model = graph.SerializeToString()
        elif kind == "vae":
            self.int_env()
            options.add_provider_for_devices(self.devices, {
                **cache, "compile_fusion_rt": "0", "onnx_custom_ops_const_key": "",
                "external_data_file": str(path.with_suffix(".pb.bin")), "hybrid_opt_token_backend": "npu",
            })
        else:
            raise ValueError(f"Unknown session kind: {kind}")
        options.register_custom_ops_library(str(self.dll))
        return ort.InferenceSession(model, options)

