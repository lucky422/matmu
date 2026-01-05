cat > flux_opt_test.py <<'PY'
import ctypes
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SO_PATH = os.path.join(os.path.dirname(__file__), "cpu.so")
MODEL_NAME = "facebook/opt-125m"
DTYPE = torch.float32

# ----------------------------
# Load Flux kernel
# ----------------------------
lib = ctypes.CDLL(SO_PATH)
flux_matmul = lib._mlir_ciface_matmul
flux_matmul.restype = None

class MemRef2D(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_longlong),
        ("sizes", ctypes.c_longlong * 2),
        ("strides", ctypes.c_longlong * 2),
    ]

def as_memref_2d(t: torch.Tensor) -> MemRef2D:
    t = t.contiguous()
    return MemRef2D(
        ctypes.c_void_p(t.data_ptr()),
        ctypes.c_void_p(t.data_ptr()),
        ctypes.c_longlong(0),
        (ctypes.c_longlong * 2)(t.shape[0], t.shape[1]),
        (ctypes.c_longlong * 2)(t.stride(0), t.stride(1)),
    )

def flux_linear_fp32(x, w):
    B, Kx = x.shape
    Kw, N = w.shape
    print(f"[flux] x={tuple(x.shape)} w={tuple(w.shape)}")
    assert Kx == Kw

    y = torch.empty((B, N), dtype=DTYPE)
    flux_matmul(
        ctypes.byref(as_memref_2d(x)),
        ctypes.byref(as_memref_2d(w)),
        ctypes.byref(as_memref_2d(y)),
    )
    return y

def sanity_check():
    A = torch.randn(128, 256, dtype=DTYPE)
    B = torch.randn(256, 128, dtype=DTYPE)
    ref = A @ B
    out = flux_linear_fp32(A, B)
    err = (out - ref).abs().max().item()
    print("[sanity] max error:", err)
    assert err < 1e-3

def patch_one_layer(model):
    layer = model.model.decoder.layers[0]
    W = layer.fc1.weight
    b = layer.fc1.bias

    def forward(x):
        if x.dim() == 3:
            B, S, K = x.shape
            y = flux_linear_fp32(
                x.reshape(B * S, K),
                W.t(),
            ).reshape(B, S, -1)
        else:
            y = flux_linear_fp32(x, W.t())
        if b is not None:
            y = y + b
        return y

    layer.fc1.forward = forward
    print("[patch] layer0.fc1 replaced")

def main():
    sanity_check()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
    ).eval()

    patch_one_layer(model)

    inp = tok("Hello Flux", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=6,
            do_sample=False,
            use_cache=False,
        )
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
PY
