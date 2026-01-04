import ctypes
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Load Flux matmul
# ----------------------------
lib = ctypes.CDLL("./cpu.so")
flux_matmul = lib._mlir_ciface_matmul
flux_matmul.restype = None

# MLIR memref descriptor
class MemRef2D(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_longlong),
        ("sizes", ctypes.c_longlong * 2),
        ("strides", ctypes.c_longlong * 2),
    ]

def as_memref(t: torch.Tensor):
    t = t.contiguous()
    return MemRef2D(
        t.data_ptr(),
        t.data_ptr(),
        0,
        (ctypes.c_longlong * 2)(*t.shape),
        (ctypes.c_longlong * 2)(t.stride(0), t.stride(1)),
    )

def flux_linear(x, w):
    # x: [B, K], w: [K, N]
    y = torch.empty((x.shape[0], w.shape[1]), dtype=torch.float16)
    flux_matmul(
        ctypes.byref(as_memref(x)),
        ctypes.byref(as_memref(w)),
        ctypes.byref(as_memref(y)),
    )
    return y

# ----------------------------
# Load tiny model
# ----------------------------
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)
model.eval()

# ----------------------------
# Monkey-patch ONE layer
# ----------------------------
orig_linear = model.model.decoder.layers[0].fc1

def flux_fc1(x):
    return flux_linear(x, orig_linear.weight.T)

model.model.decoder.layers[0].fc1.forward = flux_fc1

# ----------------------------
# Generate tokens
# ----------------------------
prompt = "Hello Flux"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=12,
        do_sample=False,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
