"""
vnnlib_utils.py
----------------
Functions for converting segments to perturbation bounds and writing VNNLIB files.

Inputs:
    - image_np : H×W×C float image, normalized in [0,1]
    - segmentation mask (H×W boolean)
    - eps : float perturbation radius
    - dilation radius : int
    - label : ground-truth class
    - num_classes : total number of classes

Outputs:
    - (lb_flat, ub_flat): flattened input bounds for each pixel
    - statistical dictionaries describing how many inputs changed
    - .vnnlib files encoding verification constraints

"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
from collections import deque

def normalize_bounds_hwc(
    lb_flat: np.ndarray,
    ub_flat: np.ndarray,
    mean: List[float],
    std: List[float],
    shape_hwc: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize flat HWC bounds with per-channel mean/std.
    Assumes std > 0 for all channels.
    """
    lb = np.asarray(lb_flat, dtype=np.float64).reshape(shape_hwc)
    ub = np.asarray(ub_flat, dtype=np.float64).reshape(shape_hwc)
    if lb.shape != ub.shape:
        raise ValueError(f"lb/ub shapes must match, got {lb.shape} vs {ub.shape}")
    if lb.shape[-1] != len(mean) or lb.shape[-1] != len(std):
        raise ValueError(
            f"Channel mismatch: shape {lb.shape}, mean {len(mean)}, std {len(std)}"
        )

    mean_arr = np.array(mean, dtype=np.float64).reshape(1, 1, -1)
    std_arr = np.array(std, dtype=np.float64).reshape(1, 1, -1)

    lb_n = (lb - mean_arr) / std_arr
    ub_n = (ub - mean_arr) / std_arr
    return lb_n.reshape(-1), ub_n.reshape(-1)

def reorder_bounds_hwc_to_chw(
    lb_flat: np.ndarray,
    ub_flat: np.ndarray,
    shape_hwc: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder flat HWC bounds to CHW order to match common ONNX input layouts.
    """
    lb = np.asarray(lb_flat, dtype=np.float64).reshape(shape_hwc)
    ub = np.asarray(ub_flat, dtype=np.float64).reshape(shape_hwc)
    if lb.shape != ub.shape:
        raise ValueError(f"lb/ub shapes must match, got {lb.shape} vs {ub.shape}")
    lb_c = np.transpose(lb, (2, 0, 1)).reshape(-1)
    ub_c = np.transpose(ub, (2, 0, 1)).reshape(-1)
    return lb_c, ub_c

def mask_to_224(mask_np: np.ndarray) -> np.ndarray:
    # mask_np: HxW boolean or 0/1
    m = Image.fromarray(mask_np.astype(np.uint8) * 255)
    m = m.resize((224, 224), Image.NEAREST)  # IMPORTANT: keep mask crisp
    return (np.asarray(m) > 127)

def report_changed_inputs(
    lb_flat: np.ndarray,
    ub_flat: np.ndarray,
    tag: str = "",
    extra: Optional[Dict] = None,
) -> Dict:
    """
    Compute how many input dimensions are allowed to change (lb != ub)
    vs how many are fixed (lb == ub). Also returns percentage.
    """
    assert lb_flat.shape == ub_flat.shape
    total = lb_flat.size
    changed_mask = np.abs(ub_flat - lb_flat) > 1e-9
    num_changed = int(changed_mask.sum())
    num_fixed = total - num_changed
    percent_changed = 100.0 * num_changed / total if total > 0 else 0.0

    # prefix = f"[VNNLIB][{tag}] " if tag else "[VNNLIB] "
    # print(
    #     f"{prefix}Inputs changed: {num_changed}, "
    #     f"fixed: {num_fixed}, total: {total}, "
    #     f"percent_changed={percent_changed:.2f}%"
    # )

    row = {
        "tag": tag,
        "num_changed": num_changed,
        "num_fixed": num_fixed,
        "total": total,
        "percent_changed": percent_changed,
    }
    if extra:
        row.update(extra)
    return row



def bounds_for_segment(
    image_np: np.ndarray,
    eps: float,
    mask: Optional[np.ndarray] = None,
    max_pixels: Optional[int] = None,
    select: str = "random",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create (lb, ub) bounds with an optional *pixel budget*.

    Semantics:
      - If ``mask`` is provided: pixels where mask==True are eligible to change.
        Else: all pixels are eligible.
      - If ``max_pixels`` is provided: at most that many *spatial* pixels (H×W)
        are allowed to change. (All 3 RGB channels for a chosen pixel change.)

    Notes:
      - ``image_np`` must be H×W×C in [0,1] (unnormalized).
      - ``select`` currently supports: "random".
    """

    img = np.asarray(image_np, dtype=np.float32)
    if img.ndim != 3:
        raise ValueError(f"image_np must be 3D (HWC); got shape {img.shape}")
    H, W, C = img.shape
    if C != 3:
        # Not strictly required, but this matches your VGG16 pipeline.
        raise ValueError(f"Expected 3 channels (RGB); got C={C}")

    mask_bool = np.ones((H, W), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask_bool.shape != (H, W):
        raise ValueError(f"mask shape {mask_bool.shape} must match spatial dims {(H, W)}")

    if max_pixels is not None:
        max_pixels = int(max_pixels)
        if max_pixels < 0:
            raise ValueError("max_pixels must be >= 0")
        ys, xs = np.where(mask_bool)
        n = int(ys.size)

        if n > max_pixels:
            if select != "random":
                raise ValueError(f"Unknown select='{select}'. Only 'random' is implemented.")
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=max_pixels, replace=False)
            limited = np.zeros((H, W), dtype=bool)
            limited[ys[idx], xs[idx]] = True
            mask_bool = limited

    eps_tensor = np.where(mask_bool[:, :, None], float(eps), 0.0).astype(np.float32)
    lb = img - eps_tensor
    ub = img + eps_tensor
    return lb.reshape(-1), ub.reshape(-1)

from pathlib import Path
import numpy as np
from typing import Optional

def write_vnnlib_for_segment(
    lb: np.ndarray,
    ub: np.ndarray,
    label: int,
    num_classes: int,
    out_path: Path,
    target_label: Optional[int] = None,
    spec_type: str = "targeted",   # "targeted" (old) or "untargeted" (CIFAR100 style)
):
    """
    Write a VNNLIB file.

    Input constraints:
        For each i: (assert (>= X_i lb_i)) and (assert (<= X_i ub_i))

    Output constraints:
        - targeted:   (assert (>= Y_target Y_label))     # your current style
        - untargeted: (assert (or (and (>= Y_j Y_label)) for all j != label))
                      # CIFAR100 VNN-COMP style: there exists some competing class >= true class
    """
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    if lb.shape != ub.shape:
        raise ValueError(f"lb/ub must match shape, got {lb.shape} vs {ub.shape}")

    n_in = int(lb.size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(x: float) -> str:
        # stable numeric formatting (no crazy scientific notation)
        return f"{float(x):.15g}"

    lines = []
    lines.append("; VNNLIB generated by pipeline")
    lines.append(f"; n_in={n_in}, num_classes={num_classes}")
    lines.append("")

    # Declare input variables
    for i in range(n_in):
        lines.append(f"(declare-const X_{i} Real)")
    lines.append("")

    # Declare output variables
    for j in range(num_classes):
        lines.append(f"(declare-const Y_{j} Real)")
    lines.append("")

    # Input constraints
    lines.append("; Input constraints:")
    for i in range(n_in):
        lines.append(f"(assert (>= X_{i} {fmt(lb[i])}))")
        lines.append(f"(assert (<= X_{i} {fmt(ub[i])}))")
        lines.append("")
    lines.append("")

    # Output constraints
    lines.append("; Output constraints:")

    if spec_type == "targeted":
        if target_label is None:
            raise ValueError("target_label is required for spec_type='targeted'")
        lines.append("; Targeted misclassification")
        lines.append(f"(assert (>= Y_{int(target_label)} Y_{int(label)}))")

    elif spec_type == "untargeted":
        # CIFAR100 VNN-COMP style:
        # adversarial exists if any other logit >= true logit
        lines.append("(assert (or")
        for j in range(num_classes):
            if j == int(label):
                continue
            lines.append(f"  (and (>= Y_{j} Y_{int(label)}))")
        lines.append("))")

    else:
        raise ValueError(f"Unknown spec_type='{spec_type}'")

    out_path.write_text("\n".join(lines), encoding="utf-8")
