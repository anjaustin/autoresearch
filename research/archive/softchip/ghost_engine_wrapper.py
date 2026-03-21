import ctypes
import torch
import numpy as np
from pathlib import Path


class GhostEngine:
    def __init__(self, model, tokenizer, base_seed, ghost_scales_path):
        self.lib = ctypes.CDLL(str(Path(__file__).parent / "ghost_engine.so"))

        class ExpertData(ctypes.Structure):
            _fields_ = [
                ("u", ctypes.POINTER(ctypes.c_float)),
                ("v", ctypes.POINTER(ctypes.c_float)),
                ("scale", ctypes.POINTER(ctypes.c_float)),
            ]

        class GhostModel(ctypes.Structure):
            _fields_ = [
                ("k_cache", ctypes.POINTER(ctypes.c_float)),
                ("v_cache", ctypes.POINTER(ctypes.c_float)),
                ("embeddings", ctypes.POINTER(ctypes.c_float)),
                ("lm_head", ctypes.POINTER(ctypes.c_float)),
                ("weight_scales", ctypes.POINTER(ctypes.c_float)),  # Order matched to C
                ("base_seed", ctypes.c_uint64),
                ("experts", ctypes.POINTER(ExpertData)),
            ]

        self.GhostModel = GhostModel
        self.ExpertData = ExpertData

        group_size = 4
        self.kv_size = 30 * group_size * 4096 * 5 * 128
        self.k_cache = torch.zeros(self.kv_size, dtype=torch.float32)
        self.v_cache = torch.zeros(self.kv_size, dtype=torch.float32)

        self.embeddings = model.get_input_embeddings().weight.data.float().contiguous()
        self.lm_head = model.lm_head.weight.data.float().contiguous()

        scales_dict = torch.load(ghost_scales_path, weights_only=True)
        ordered_scales = []
        layer_names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        for i in range(30):
            for ln in layer_names:
                full_name = f"model.layers.{i}.{ln}"
                ordered_scales.append(scales_dict[full_name])
        self.weight_scales = torch.tensor(
            ordered_scales, dtype=torch.float32
        ).contiguous()

        self.experts_array = (ExpertData * 840)()

        self.model_struct = GhostModel()
        self.model_struct.k_cache = ctypes.cast(
            self.k_cache.data_ptr(), ctypes.POINTER(ctypes.c_float)
        )
        self.model_struct.v_cache = ctypes.cast(
            self.v_cache.data_ptr(), ctypes.POINTER(ctypes.c_float)
        )
        self.model_struct.embeddings = ctypes.cast(
            self.embeddings.data_ptr(), ctypes.POINTER(ctypes.c_float)
        )
        self.model_struct.lm_head = ctypes.cast(
            self.lm_head.data_ptr(), ctypes.POINTER(ctypes.c_float)
        )
        self.model_struct.weight_scales = ctypes.cast(
            self.weight_scales.data_ptr(), ctypes.POINTER(ctypes.c_float)
        )
        self.model_struct.base_seed = base_seed
        self.model_struct.experts = ctypes.cast(
            self.experts_array, ctypes.POINTER(ExpertData)
        )

        self.lib.ghost_engine_generate_batched.argtypes = [
            ctypes.POINTER(GhostModel),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]

    def rollout_batch(self, adapters, prompt_ids, group_size, gen_len, temp=0.7):
        # u, v, and scale are stored as BF16 in GhostExpert but the C engine
        # expects float32 pointers.  Convert and keep the tensors alive for the
        # duration of the C call; letting them go out of scope would invalidate
        # the pointers before ghost_engine_generate_batched returns.
        self._f32_buf = []
        for i, (name, expert) in enumerate(adapters):
            u_f32 = expert.u.float().contiguous()
            v_f32 = expert.v.float().contiguous()
            s_f32 = expert.scale.float().contiguous()
            self._f32_buf.extend([u_f32, v_f32, s_f32])
            self.experts_array[i].u = ctypes.cast(
                u_f32.data_ptr(), ctypes.POINTER(ctypes.c_float)
            )
            self.experts_array[i].v = ctypes.cast(
                v_f32.data_ptr(), ctypes.POINTER(ctypes.c_float)
            )
            self.experts_array[i].scale = ctypes.cast(
                s_f32.data_ptr(), ctypes.POINTER(ctypes.c_float)
            )

        prompt_len = prompt_ids.shape[1]
        total_len = prompt_len + gen_len
        tokens = np.zeros(group_size * total_len, dtype=np.int32)

        for g in range(group_size):
            tokens[g * total_len : g * total_len + prompt_len] = prompt_ids[0].numpy()

        tokens_ptr = tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.lib.ghost_engine_generate_batched(
            ctypes.byref(self.model_struct),
            tokens_ptr,
            group_size,
            prompt_len,
            gen_len,
            temp,
        )

        return torch.from_numpy(tokens).reshape(group_size, total_len)
