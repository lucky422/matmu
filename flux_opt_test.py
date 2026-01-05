cat > flux_opt_test.py <<'PY'
import ctypes
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Config
# ----------------------------
SO_PATH = os.path.join(os.path.dirname(__file__), "cpu.so")
MODEL_NAME = "facebook/opt-125m"

# IMPORTANT: We run FP32 on CPU for safety first.
DTYPE = torch.float32

# ----------------------------
# Load Flux .so
# ----------------------------
lib = ctypes.CDLL(SO_PATH)
flux_matmul = lib._mlir_ciface_matmul
flux_matmul.restype = None

# MLIR memref descriptor for rank-2 tensor
class MemRef2D(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_longlong),
        ("sizes", ctypes.c_longlong * 2),
        ("strides", ctypes.c_longlong * 2),
    ]

def as_memref_2d(t: torch.Tensor) -> MemRef2D:
    # Must be contiguous for our first integration proof.
    t = t.contiguous()
    assert t.dim() == 2, f"Expected rank-2 tensor, got shape {tuple(t.shape)}"
    assert t.dtype == DTYPE, f"Expected dtype {DTYPE}, got {t.dtype}"
    return MemRef2D(
        ctypes.c_void_p(t.data_ptr()),
        ctypes.c_void_p(t.data_ptr()),
        ctypes.c_longlong(0),
        (ctypes.c_longlong * 2)(ctypes.c_longlong(t.shape[0]), ctypes.c_longlong(t.shape[1])),
        (ctypes.c_longlong * 2)(ctypes.c_longlong(t.stride(0)), ctypes.c_longlong(t.stride(1))),
    )

def flux_linear_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    x: [B, K]  fp32 contiguous
    w: [K, N]  fp32 contiguous
    returns y: [B, N] fp32
    """
    x = x.to(dtype=DTYPE).contiguous()
    w = w.to(dtype=DTYPE).contiguous()

    B, Kx = x.shape
    Kw, N = w.shape
    assert Kx == Kw, f"K mismatch: x is {Kx}, w is {Kw}"

    y = torch.empty((B, N), dtype=DTYPE)

    # Call MLIR C-interface: (_mlir_ciface_matmul)(memrefA*, memrefB*, memrefC*)
    flux_matmul(
        ctypes.byref(as_memref_2d(x)),
        ctypes.byref(as_memref_2d(w)),
        ctypes.byref(as_memref_2d(y)),
    )
    return y

def sanity_check_kernel():
    """
    Before touching transformers, prove the .so does correct math for one fixed shape.
    This avoids debugging model stuff if kernel ABI is wrong.
    """
    torch.manual_seed(0)
    A = torch.randn(128, 256, dtype=DTYPE)
    B = torch.randn(256, 128, dtype=DTYPE)
    Y_ref = A @ B
    Y = flux_linear_fp32(A, B)
    max_abs = (Y - Y_ref).abs().max().item()
    print("[sanity] max abs error:", max_abs)
    # fp32 should be very close
    assert max_abs < 1e-3, "Kernel sanity check failed (too much error)."

def patch_one_layer(model):
    """
    Patch ONE linear (fc1) in layer0 to use flux matmul.
    We keep everything else as PyTorch, to minimize crash surface area.
    """
    layer0 = model.model.decoder.layers[0]
    orig_fc1 = layer0.fc1

    # OPT fc1 expects x @ W^T + bias; weight stored as [out, in]
    W = orig_fc1.weight  # [out, in]
    b = orig_fc1.bias

    def flux_fc1_forward(x):
        # x: [B, S, in] or [B*S, in] depending on callsite
        x_in = x
        if x_in.dim() == 3:
            B, S, K = x_in.shape
            x2 = x_in.reshape(B * S, K).contiguous()
            y2 = flux_linear_fp32(x2, W.t())  # [B*S, out]
            if b is not None:
                y2 = y2 + b.to(dtype=DTYPE)
            return y2.reshape(B, S, -1)
        elif x_in.dim() == 2:
            y = flux_linear_fp32(x_in, W.t())
            if b is not None:
                y = y + b.to(dtype=DTYPE)
            return y
        else:
            raise RuntimeError(f"Unexpected fc1 input rank: {x_in.dim()}")

    layer0.fc1.forward = flux_fc1_forward
    print("[patch] layer0.fc1 now uses Flux matmul")

def main():
    # 1) Prove cpu.so is correct BEFORE model.
    sanity_check_kernel()

    # 2) Load model in FP32 CPU mode.
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
    model.eval()

    # 3) Patch one layer only.
    patch_one_layer(model)

    # 4) Run minimal generation (disable cache to reduce shape dynamism).
    prompt = "Hello Flux"
    inputs = tok(prompt, return_tensors="pt")
    assert inputs["input_ids"].shape[0] == 1, "Batch must be 1 for this test."

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=False,
        )

    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
PY
