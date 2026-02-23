"""
pipeline.py
-------------
Main entrypoint for the segmentation → bounds → VNNLIB generation pipeline.

Inputs:
    - Command-line args:
        • --config path/to/config.yaml
        • --indices N1 N2 ...
        • --images path1 path2 ...

Outputs:
    - vnnlib/ folder with generated property files
    - debug_vis/ folder with segment overlays
    - instances.csv listing (onnx_path, vnnlib_path, timeout)
    - input_change_stats.csv reporting semantic coverage per segment

Purpose:
    Orchestrates the full pipeline:
        For each selected image:
            - run segmentation
            - compute bounds
            - write VNNLIBs
            - log stats
        Write CSV outputs
"""

import argparse
import csv
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import onnxruntime as ort
import json
import numpy as np

from os import path
from pathlib import Path, PosixPath
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor
from config_utils import load_config, ensure_dir
from dataset_utils import load_imagenet_vggnet16_metadata,get_imagenet_label_for_image,load_cifar100_decoded_metadata, get_cifar_label_for_image
from sam2_utils import load_sam2_predictor, run_segmentation_model
from vnnlib_utils import bounds_for_segment,report_changed_inputs,write_vnnlib_for_segment, normalize_bounds_hwc, reorder_bounds_hwc_to_chw
from vis_utils import visualize_segments, visualize_selected_masks_on_image
from collections import deque
from torchvision.transforms.functional import InterpolationMode

VGG16_MEAN = [0.485, 0.456, 0.406]
VGG16_STD = [0.229, 0.224, 0.225]

CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD  = [0.2673, 0.2564, 0.2761]

import re

MODEL_KEY_RE = re.compile(r"^(?P<key>[^_]+(?:_[^_]+)?)__")  
# matches: "resnet_large__..." or "resnet_medium__..."

def infer_model_key_from_image_path(img_path: Path) -> str:
    """
    Infer model key from filename prefix:
      resnet_large__label_10__idx_8477.png -> "resnet_large"
    """
    m = MODEL_KEY_RE.match(img_path.name)
    if not m:
        raise ValueError(
            f"Cannot infer model key from filename: {img_path.name}\n"
            f"Expected prefix like 'resnet_large__' or 'resnet_medium__'."
        )
    return m.group("key")

def build_onnx_session_map(cfg: Dict, onnx_dir: Path) -> Dict[str, Dict]:
    """
    Builds sessions for the ONNX models in cfg["onnx"]["target_name"] but does NOT loop
    over them for every image. Instead, returns a dict keyed by model key, e.g.
      {
        "resnet_large": {"session": ..., "rel_path": "onnx/CIFAR100_resnet_large.onnx", "model_tag": "CIFAR100_resnet_large"},
        "resnet_medium": {...}
      }
    The mapping from model key -> target_name uses substring matching.
    """
    targets = cfg["onnx"].get("target_name", None)
    if not isinstance(targets, (list, tuple)):
        targets = [targets] if targets is not None else []

    # Copy + build sessions using your existing resolver (it handles src dir/file)
    entries = resolve_onnx_targets(cfg, onnx_dir)

    # Build a session map by heuristics on the ONNX filename
    session_map: Dict[str, Dict] = {}

    for e in entries:
        name = Path(e["rel_path"]).name.lower()

        if "large" in name:
            session_map["resnet_large"] = e
        elif "medium" in name:
            session_map["resnet_medium"] = e
        else:
            # fallback key: stem
            session_map[Path(name).stem] = e

    if not session_map:
        raise RuntimeError("No ONNX sessions created; check onnx.source_model/target_name in config.")

    print("[ONNX] Session map keys:", sorted(session_map.keys()))
    return session_map

def cifar_preprocess(
    img_pil,
    normalize=True,
    size=32,
    antialias=False,
    skip_resize_if_exact=True,
):
    """
    CIFAR decoded images may already be 32x32; we still enforce size via resize+center_crop.
    Returns CHW tensor in [0,1] (or normalized).
    """
    if skip_resize_if_exact and img_pil.size == (size, size):
        img_r = img_pil
    else:
        img_r = F.resize(
            img_pil,
            size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=antialias,
        )
    img_c = F.center_crop(img_r, [size, size])
    img_t = F.to_tensor(img_c)  # [3,H,W] in [0,1]
    if normalize:
        img_t = F.normalize(img_t, mean=CIFAR100_MEAN, std=CIFAR100_STD)
    return img_t

def resolve_onnx_targets(cfg: Dict, onnx_dir: Path):
    """
    Supports:
      - cfg["onnx"]["source_model"] = /path/to/model.onnx   (single file)
      - cfg["onnx"]["source_model"] = /path/to/dir          (directory)
        cfg["onnx"]["target_name"]  = ["A.onnx","B.onnx"]   (list)
    Returns list of dicts: [{"session":..., "rel_path":..., "model_tag":...}, ...]
    """
    src = Path(cfg["onnx"]["source_model"])
    target = cfg["onnx"].get("target_name", None)

    # Normalize target_name into a list
    if target is None:
        # fallback: if src is file, use its name; if dir, error
        if src.is_file():
            target_names = [src.name]
        else:
            raise ValueError(
                "onnx.target_name is required when onnx.source_model is a directory."
            )
    elif isinstance(target, (list, tuple)):
        target_names = list(target)
    else:
        target_names = [str(target)]

    ort.set_default_logger_severity(3)

    out = []
    for name in target_names:
        if src.is_dir():
            src_file = src / name
        else:
            # if src is a file, allow either exact match or ignore name mismatch and copy src
            src_file = src

        if not src_file.exists():
            raise FileNotFoundError(f"[ONNX] Missing ONNX file: {src_file}")

        dst_file = onnx_dir / name
        if not dst_file.exists():
            print(f"[ONNX] Copying {src_file} -> {dst_file}")
            dst_file.write_bytes(src_file.read_bytes())
        else:
            print(f"[ONNX] Using existing {dst_file}")

        sess = ort.InferenceSession(str(dst_file))
        rel = str(Path("onnx") / dst_file.name)

        model_tag = Path(name).stem  # e.g. "CIFAR100_resnet_large"
        out.append({"session": sess, "rel_path": rel, "model_tag": model_tag})

    return out

def make_ratio_pixel_selection_mask(
    seg_mask: np.ndarray,
    k: int,
    seed: int = 0,
    select: str = "random",
):
    """
    Returns three boolean masks:
      1) mask_global : all selected pixels (exactly k True)
      2) mask_in     : selected pixels where seg_mask == 1
      3) mask_out    : selected pixels where seg_mask == 0

    select:
        "random"       -> uniform random pixels (old behaviour)
        "concentrated" -> spatially contiguous blocks in each region
    """

    m = np.asarray(seg_mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"seg_mask must be 2D, got {m.shape}")

    H, W = m.shape
    total = H * W
    k = int(k)

    if k <= 0:
        z = np.zeros((H, W), dtype=bool)
        return z, z.copy(), z.copy()

    if k >= total:
        ones = np.ones((H, W), dtype=bool)
        return ones, ones & m, ones & ~m

    n_in = int(m.sum())
    n_out = total - n_in

    # --------------------------------------------------
    # Decide ratio
    # --------------------------------------------------
    if n_in == 0:
        k_in, k_out = 0, k
    elif n_out == 0:
        k_in, k_out = k, 0
    else:
        frac_in = n_in / total
        k_in = int(round(k * frac_in))
        k_in = min(k_in, n_in)
        k_out = k - k_in

        if k_out > n_out:
            k_out = n_out
            k_in = k - k_out
        if k_in > n_in:
            k_in = n_in
            k_out = k - k_in

    rng = np.random.default_rng(seed)

    mask_in = np.zeros((H, W), dtype=bool)
    mask_out = np.zeros((H, W), dtype=bool)

    # ==================================================
    # RANDOM MODE  (old behaviour)
    # ==================================================
    if select == "random":

        if k_in > 0:
            ys, xs = np.where(m)
            idx = rng.choice(len(ys), size=k_in, replace=False)
            mask_in[ys[idx], xs[idx]] = True

        if k_out > 0:
            ys, xs = np.where(~m)
            idx = rng.choice(len(ys), size=k_out, replace=False)
            mask_out[ys[idx], xs[idx]] = True

    # ==================================================
    # CONCENTRATED MODE
    # ==================================================
    elif select == "concentrated":

        def grow_region(region_mask, k_target):
            """ BFS expansion inside region_mask """
            out = np.zeros((H, W), dtype=bool)

            ys, xs = np.where(region_mask)
            if len(ys) == 0 or k_target == 0:
                return out

            start = rng.integers(len(ys))
            sy, sx = ys[start], xs[start]

            q = deque([(sy, sx)])
            out[sy, sx] = True
            visited = set([(sy, sx)])

            while q and out.sum() < k_target:
                y, x = q.popleft()

                for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < H
                        and 0 <= nx < W
                        and region_mask[ny, nx]
                        and (ny, nx) not in visited
                    ):
                        visited.add((ny, nx))
                        out[ny, nx] = True
                        q.append((ny, nx))
                        if out.sum() >= k_target:
                            break

            return out

        if k_in > 0:
            mask_in = grow_region(m, k_in)

        if k_out > 0:
            mask_out = grow_region(~m, k_out)

    else:
        raise ValueError(f"Unknown select mode: {select}")

    mask_global = mask_in | mask_out
    return mask_global, mask_in, mask_out

def load_manual_prompts(prompts_path: Path) -> dict:
    prompts = {}
    for p in prompts_path.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            prompts.update(json.load(f))
    return prompts

def prompts_for_image(prompts_db: dict, image_key: str):
    """
    prompts_db example:
    {
      "myimg.jpg": {
        "foreground": [[x,y], [x,y]],
        "background": [[x,y]]
      }
    }
    """
    entry = prompts_db.get(image_key)
    if entry is None:
        return None, None

    fg = entry.get("foreground", [])
    bg = entry.get("background", [])

    # labels: 1 for FG, 0 for BG
    coords = np.array(fg + bg, dtype=np.float32)
    labels = np.array([1] * len(fg) + [0] * len(bg), dtype=np.int64)

    if coords.size == 0:
        return None, None
    return coords, labels

def vgg16_preprocess(img_pil, normalize=True, size=224):
    img_r = F.resize(img_pil, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
    img_c = F.center_crop(img_r, [size, size])
    img_t = F.to_tensor(img_c)  # [3,H,W] in [0,1]
    if normalize:
        img_t = F.normalize(img_t, mean=VGG16_MEAN, std=VGG16_STD)
    return img_t

def debug_print_mask_binary(mask_t: torch.Tensor, tag: str = ""):
    """
    Prints a binary sanity check (0/1) in percentage.
    mask_t is expected bool or {0,1}. Shape [1,H,W] or [H,W].
    """
    if mask_t.ndim == 3:
        m = mask_t[0]
    else:
        m = mask_t

    # Convert to int for unique/value checks
    m_int = m.to(torch.int32)
    uniq = torch.unique(m_int).cpu().tolist()

    total = m.numel()
    ones = int(m.sum().item())
    zeros = total - ones
    ones_pct = 100.0 * ones / total if total else 0.0
    zeros_pct = 100.0 * zeros / total if total else 0.0

    print(f"[DEBUG][MASK]{'['+tag+']' if tag else ''} unique={uniq} "
          f"ones={ones_pct:.2f}% zeros={zeros_pct:.2f}% (H={m.shape[-2]}, W={m.shape[-1]})")

def save_preprocess_debug_vis(
    out_dir: Path,
    prefix: str,
    img_pil_before: Image.Image,
    mask_pil_before: Image.Image,
    img_t_after: torch.Tensor,
    mask_t_after: torch.Tensor,
):
    """
    Saves:
      - before_mask_bw.png
      - before_overlay.png
      - after_mask_bw.png
      - after_overlay.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- BEFORE: convert to numpy ---
    img_before = np.asarray(img_pil_before).astype(np.float32) / 255.0
    mask_before = np.asarray(mask_pil_before)
    if mask_before.ndim == 3:
        mask_before = mask_before[..., 0]
    mask_before = (mask_before > 0).astype(np.float32)

    # --- AFTER: de-normalize image tensor back to [0,1] for visualization ---
    img_after = img_t_after.detach().cpu().clone()
    for c in range(3):
        img_after[c] = img_after[c] * VGG16_STD[c] + VGG16_MEAN[c]
    img_after = img_after.clamp(0, 1).permute(1, 2, 0).numpy()

    if mask_t_after.ndim == 3:
        m_after = mask_t_after[0].detach().cpu().numpy().astype(np.float32)
    else:
        m_after = mask_t_after.detach().cpu().numpy().astype(np.float32)

    # --- Save BEFORE mask BW ---
    plt.figure()
    plt.imshow(mask_before, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_before_mask_bw.png", dpi=200)
    plt.close()

    # --- Save BEFORE overlay ---
    plt.figure()
    plt.imshow(img_before)
    plt.imshow(mask_before, cmap="gray", alpha=0.5, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_before_overlay.png", dpi=200)
    plt.close()

    # --- Save AFTER mask BW ---
    plt.figure()
    plt.imshow(m_after, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_after_mask_bw.png", dpi=200)
    plt.close()

    # --- Save AFTER overlay ---
    plt.figure()
    plt.imshow(img_after)
    plt.imshow(m_after, cmap="gray", alpha=0.5, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_after_overlay.png", dpi=200)
    plt.close()

def process_single_image(
    img_path: Path,
    predictor: SAM2ImagePredictor,
    cfg: Dict,
    get_label_fn: Callable[[Path], int],
    onnx_session: ort.InferenceSession,
    onnx_rel_path: str,
    model_tag: str = "model",
    preprocess_for_onnx: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    preprocess_for_seg: Optional[Callable[[Image.Image], np.ndarray]] = None,
    vnnlib_dir: Optional[Path] = None,
    debug_dir: Optional[Path] = None,
    stats_rows: Optional[List[Dict]] = None,
    vnn_spec_type: str = "targeted",
) -> List[Tuple[str, str, int]]:
    """
    Returns list of instances.csv rows for this image.
    Each row: (onnx_rel_path, vnnlib_rel_path, timeout)
    """
    print("\n==============================")
    print(f"Processing image: {img_path}")
    timeout = int(cfg["output"]["timeout"])
    epsilons = list(cfg["verification"]["epsilon"])
    num_classes = int(cfg["verification"]["num_classes"])

    dataset_type = cfg.get("dataset", {}).get("type", "imagenet_vgg16")
    image = Image.open(img_path).convert("RGB")
    image.load()  # force decode now for determinism

    # Backward-compatible fallbacks (in case an older call site still exists)
    if preprocess_for_onnx is None or preprocess_for_seg is None:
        dataset_type = cfg.get("dataset", {}).get("type", "imagenet_vgg16")
        if dataset_type == "imagenet_vgg16":
            preprocess_for_onnx = lambda im: vgg16_preprocess(im, normalize=True, size=224)
            preprocess_for_seg  = lambda im: vgg16_preprocess(im, normalize=False, size=224).permute(1, 2, 0).numpy()
        elif dataset_type == "cifar100":
            preprocess_for_onnx = lambda im: cifar_preprocess(
                im, normalize=True, size=32, antialias=False, skip_resize_if_exact=True
            )
            preprocess_for_seg  = lambda im: cifar_preprocess(
                im, normalize=False, size=32, antialias=False, skip_resize_if_exact=True
            ).permute(1, 2, 0).numpy()
        else:
            raise ValueError(f"Unknown dataset.type='{dataset_type}'")

    if vnnlib_dir is None:
        raise ValueError("vnnlib_dir is required (pass it from main())")

    # --- ONNX input (dataset-aware) ---
    img_t = preprocess_for_onnx(image)                 # CHW, normalized for model
    image_np_chw = img_t.detach().cpu().numpy()        # (3,H,W)

    # Verify the image is correctly classified before generating specs
    label = get_label_fn(img_path)
    print(f"True label index: {label} / {num_classes}")

    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_input = np.expand_dims(image_np_chw, axis=0).astype(np.float32, copy=False)
    logits = onnx_session.run(None, {onnx_input_name: onnx_input})[0]
    pred_label = int(np.argmax(logits, axis=1)[0])
    print(f"Predicted label index: {pred_label} / {num_classes}")
    if pred_label != label:
        print(f"[SKIP] Model top-1 ({pred_label}) != ground truth ({label}); skipping image.")
        return []

    # Runner-up target label (benchmark style)
    logits_flat = logits[0].reshape(-1)
    order = np.argsort(logits_flat)
    top1_label = int(order[-1])
    runner_up_label = int(order[-2]) if order.size >= 2 else label
    if top1_label != label:
        print(f"[WARN] Top-1 from logits ({top1_label}) != label ({label}); using label as top-1.")
        top1_label = label

    # --- Segmentation input (unnormalized [0,1], HWC) ---
    image_np_hwc = preprocess_for_seg(image)  # HWC float32 in [0,1]
    image_np_hwc_seg = image_np_hwc

    ver_cfg = cfg.get("verification", {})
    pixel_counts = ver_cfg.get("perturb_pixels", [None])
    if isinstance(pixel_counts, int):
        pixel_counts = [pixel_counts]
    select = ver_cfg.get("perturb_select", "random")
    seed = int(ver_cfg.get("perturb_seed", 0))

    seg_cfg = cfg["segmentation"]
    point_coords = None
    point_labels = None

    if seg_cfg.get("prompt_mode", "grid") == "manual":
        prompts_path = Path(seg_cfg["prompts_path"])
        prompts_db = load_manual_prompts(prompts_path)
        coords, labels2 = prompts_for_image(prompts_db, Path(img_path).name)
        if coords is None:
            print(f"[PROMPTS] No manual prompts found for {Path(img_path).name}; falling back to grid.")
        else:
            point_coords, point_labels = coords, labels2

    segments = run_segmentation_model(
        predictor=predictor,
        image_np=image_np_hwc_seg,
        grid_size=seg_cfg.get("grid_size", 6),
        iou_threshold=seg_cfg.get("iou_threshold", 0.8),
        score_threshold=seg_cfg.get("score_threshold", 0.0),
        max_segments=seg_cfg.get("max_segments", None),
        point_coords=point_coords,
        point_labels=point_labels,
        dilation_radius=seg_cfg.get("dilation_radius", 0),
    )

    # Use the model key already baked into the image filename to keep names stable
    # Example stem: "resnet_large__label_10__idx_8477"

    # img_basename = f"{model_tag}__{img_path.stem}"
    img_basename = f"{img_path.stem}"

    if seg_cfg["vis"]:
        visualize_segments(
            image_np_hwc,
            segments,
            debug_dir,
            img_basename,
        )

    csv_rows: List[Tuple[str, str, int]] = []

    # ------------------------------------------------------
    # For each segment (plus global), write vnnlibs per eps,k
    # ------------------------------------------------------
    for eps in epsilons:
        eps = float(eps)

        # Global: all pixels eligible (or pixel-budgeted)
        for k in pixel_counts:
            lb_flat, ub_flat = bounds_for_segment(
                image_np=image_np_hwc,
                eps=eps,
                mask=None,
                max_pixels=k,
                select="random",
                seed=seed,
            )
            if dataset_type == "cifar100":
                lb_flat, ub_flat = normalize_bounds_hwc(
                    lb_flat, ub_flat, CIFAR100_MEAN, CIFAR100_STD, image_np_hwc.shape
                )
                lb_flat, ub_flat = reorder_bounds_hwc_to_chw(
                    lb_flat, ub_flat, image_np_hwc.shape
                )

            vnn_name = f"{img_basename}_global_k{k}_eps_{eps}.vnnlib"
            vnn_path = vnnlib_dir / vnn_name

            write_vnnlib_for_segment(
                lb=lb_flat,
                ub=ub_flat,
                label=label,
                num_classes=num_classes,
                out_path=vnn_path,
                target_label=runner_up_label if vnn_spec_type == "targeted" else None,
                spec_type=vnn_spec_type,
            )

            vnn_rel = str(Path("vnnlib") / vnn_name)
            csv_rows.append((onnx_rel_path, vnn_rel, timeout))

            if stats_rows is not None:
                row = report_changed_inputs(
                    lb_flat, ub_flat,
                    tag="global",
                    extra={
                        "image": img_basename,
                        "is_global": True,
                        "segment_index": -1,
                        "eps": eps,
                        "k": k,
                        "score": None,
                        "bbox": None,
                        "pattern": select,
                        "model": model_tag,
                    },
                )
                stats_rows.append(row)

        # Per-segment masks: fix_mask / fix_nonmask logic
        for sidx, seg in enumerate(segments):
            seg_mask = np.asarray(seg["mask"], dtype=bool)
            score = float(seg.get("score", 0.0))
            bbox = seg.get("bbox", None)

            # For each segment, create two threat sets:
            #  - fix_mask    : perturb outside the segment (mask==False)
            #  - fix_nonmask : perturb inside the segment (mask==True)
            for tag, mask in [
                ("fix_mask", ~seg_mask),
                ("fix_nonmask", seg_mask),
            ]:
                for k in pixel_counts:
                    # Optional: ratio-budgeted selection concentrated/random
                    # If you want the ratio split: use your make_ratio_pixel_selection_mask
                    # Here we keep your current behavior: bounds_for_segment can budget within mask.
                    lb_flat, ub_flat = bounds_for_segment(
                        image_np=image_np_hwc,
                        eps=eps,
                        mask=mask,
                        max_pixels=k,
                        select="random",
                        seed=seed,
                    )
                    if dataset_type == "cifar100":
                        lb_flat, ub_flat = normalize_bounds_hwc(
                            lb_flat, ub_flat, CIFAR100_MEAN, CIFAR100_STD, image_np_hwc.shape
                        )
                        lb_flat, ub_flat = reorder_bounds_hwc_to_chw(
                            lb_flat, ub_flat, image_np_hwc.shape
                        )

                    vnn_name = f"{img_basename}_{tag}_seg{sidx}_k{k}_eps_{eps}.vnnlib"
                    vnn_path = vnnlib_dir / vnn_name

                    write_vnnlib_for_segment(
                        lb=lb_flat,
                        ub=ub_flat,
                        label=label,
                        num_classes=num_classes,
                        out_path=vnn_path,
                        target_label=runner_up_label if vnn_spec_type == "targeted" else None,
                        spec_type=vnn_spec_type,
                    )

                    vnn_rel = str(Path("vnnlib") / vnn_name)
                    csv_rows.append((onnx_rel_path, vnn_rel, timeout))

                    if stats_rows is not None:
                        row = report_changed_inputs(
                            lb_flat, ub_flat,
                            tag=tag,
                            extra={
                                "image": img_basename,
                                "is_global": False,
                                "segment_index": sidx,
                                "eps": eps,
                                "k": k,
                                "score": score,
                                "bbox": bbox,
                                "pattern": select,
                                "model": model_tag,
                            },
                        )
                        stats_rows.append(row)

    return csv_rows

def expand_image_entries(entries: Iterable[Union[str, Path]]) -> List[Path]:
    """
    Normalize a mixed list of files/dirs into a list of image files.
    Directories are expanded to common image formats; non-dirs are passed through.
    """
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    expanded: List[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            expanded.extend(
                sorted(
                    f
                    for f in path.iterdir()
                    if f.is_file() and f.suffix.lower() in allowed_suffixes
                )
            )
        else:
            expanded.append(path)
    return expanded


def main():
    parser = argparse.ArgumentParser(
        description="Image dataset → SAM2 segmentation → VNNLIB pipeline"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        help="Dataset indices (0-based) to process",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="*",
        help="Explicit image paths to process",
    )
    # TODO: delete later
    # import sys
    # sys.argv += ["--config", "configs/cifar100/cifar100_one_img.yaml"]

    args = parser.parse_args()
    cfg = load_config(args.config)

    # -----------------------------
    # Prepare benchmark directories
    # -----------------------------
    out_root = ensure_dir(Path(cfg["output"]["benchmark_dir"]))
    onnx_dir = ensure_dir(out_root / "onnx")
    vnnlib_dir = ensure_dir(out_root / "vnnlib")
    instances_csv_path = out_root / "instances.csv"
    debug_vis_dir = ensure_dir(out_root / "debug_vis")
    stats_csv_path = out_root / "input_change_stats.csv"
    change_stats_rows: List[Dict] = []

     # -----------------------------
    # Dataset (type-aware)
    # -----------------------------
    dataset_type = cfg.get("dataset", {}).get("type", "imagenet_vggnet16")

    # CIFAR100 uses VNN-COMP style "untargeted" output constraint,
    # ImageNet/VGG keeps the old targeted style
    if dataset_type == "cifar100":
        vnn_spec_type = "untargeted"
    else:
        vnn_spec_type = "targeted"

    if dataset_type == "imagenet_vggnet16":
        all_images, img_to_label, num_classes = load_imagenet_vggnet16_metadata(cfg["dataset"])

        def get_label(img_path: Path) -> int:
            return get_imagenet_label_for_image(img_path, img_to_label)

        preprocess_for_onnx = lambda im: vgg16_preprocess(im, normalize=True, size=224)
        preprocess_for_seg  = lambda im: vgg16_preprocess(im, normalize=False, size=224).permute(1, 2, 0).numpy()

    elif dataset_type == "cifar100":
        all_images, img_to_label, num_classes = load_cifar100_decoded_metadata(cfg["dataset"])

        def get_label(img_path: Path) -> int:
            return get_cifar_label_for_image(img_path, img_to_label)

        preprocess_for_onnx = lambda im: cifar_preprocess(
            im, normalize=True, size=32, antialias=False, skip_resize_if_exact=True
        )
        preprocess_for_seg  = lambda im: cifar_preprocess(
            im, normalize=False, size=32, antialias=False, skip_resize_if_exact=True
        ).permute(1, 2, 0).numpy()

    else:
        raise ValueError(f"Unknown dataset.type='{dataset_type}' (expected 'imagenet_vgg16' or 'cifar100')")

    # no vnnlib source override

    print(f"[Dataset] Found {len(all_images)} images; num_classes = {num_classes}")
    cfg.setdefault("verification", {})
    cfg["verification"]["num_classes"] = num_classes

    # -----------------------------
    # Choose which images to process (args > config > all)
    # -----------------------------
    images_to_process: List[Path] = []

    # 1) CLI --images has highest priority
    if args.images:
        images_to_process = expand_image_entries(args.images)

    # 2) Otherwise CLI --indices
    elif args.indices:
        for idx in args.indices:
            if idx < 0 or idx >= len(all_images):
                raise IndexError(f"--indices contains {idx} but dataset has {len(all_images)} images.")
            images_to_process.append(all_images[idx])

    # 3) Otherwise config.run.image_paths / config.run.indices (optional)
    else:
        run_cfg = cfg.get("run", {})

        cfg_image_paths = run_cfg.get("image_paths", None)
        cfg_indices = run_cfg.get("indices", None)

        if cfg_image_paths:
            if isinstance(cfg_image_paths, (str, Path)):
                cfg_image_paths = [cfg_image_paths]
            images_to_process = expand_image_entries(cfg_image_paths)

        elif cfg_indices:
            for idx in cfg_indices:
                if idx < 0 or idx >= len(all_images):
                    raise IndexError(f"config.run.indices contains {idx} but dataset has {len(all_images)} images.")
                images_to_process.append(all_images[idx])

        else:
            # fallback: everything (explicit)
            images_to_process = list(all_images)

    print(f"[RUN] Will process {len(images_to_process)} image(s)")

    # -----------------------------
    # Load SAM2 model
    # -----------------------------
    predictor = load_sam2_predictor(cfg["segmentation"])

    # -----------------------------
    # Build ONNX sessions (but do NOT run both per image)
    # -----------------------------
    session_map = build_onnx_session_map(cfg, onnx_dir)

    # -----------------------------
    # Run pipeline (choose model based on image name)
    # -----------------------------
    all_rows: List[Tuple[str, str, int]] = []

    for img_path in images_to_process:
        if dataset_type == "cifar100":
            model_key = infer_model_key_from_image_path(img_path)
        else:
            if len(session_map) != 1:
                raise ValueError(
                    "Multiple ONNX sessions found, but dataset.type is not 'cifar100'.\n"
                    "Set dataset.type='cifar100' to infer model key per image, or provide a single ONNX model."
                )
            model_key = next(iter(session_map.keys()))

        if model_key not in session_map:
            raise KeyError(
                f"Model key '{model_key}' not found in ONNX session map.\n"
                f"Available: {sorted(session_map.keys())}\n"
                f"Image: {img_path}"
            )

        onnx_session = session_map[model_key]["session"]
        onnx_rel_path = session_map[model_key]["rel_path"]
        model_tag = session_map[model_key]["model_tag"]

        rows = process_single_image(
            img_path=img_path,
            predictor=predictor,
            cfg=cfg,
            get_label_fn=get_label,
            onnx_session=onnx_session,
            onnx_rel_path=onnx_rel_path,
            model_tag=model_tag,
            preprocess_for_onnx=preprocess_for_onnx,
            preprocess_for_seg=preprocess_for_seg,
            vnnlib_dir=vnnlib_dir,
            debug_dir=debug_vis_dir,
            stats_rows=change_stats_rows,
            vnn_spec_type=vnn_spec_type,
        )
        all_rows.extend(rows)

    # -----------------------------
    # Write instances.csv
    # -----------------------------
    print(f"\n[CSV] Writing {len(all_rows)} rows to {instances_csv_path}")
    with instances_csv_path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(all_rows)

    if change_stats_rows:
        fieldnames = [
            "image",
            "model",
            "tag",
            "is_global",
            "segment_index",
            "eps",
            "k",
            "score",
            "bbox",
            "num_changed",
            "num_fixed",
            "total",
            "percent_changed",
            "pattern",
        ]
        print(f"[CSV] Writing {len(change_stats_rows)} rows to {stats_csv_path}")
        with stats_csv_path.open("w", newline="") as f_stats:
            writer = csv.DictWriter(f_stats, fieldnames=fieldnames)
            writer.writeheader()
            for row in change_stats_rows:
                for key in fieldnames:
                    row.setdefault(key, None)
                writer.writerow(row)
    else:
        print("[CSV] No change-stats rows collected, not writing stats CSV.")

    print("\n[DONE] Pipeline finished.")


if __name__ == "__main__":
    main()
