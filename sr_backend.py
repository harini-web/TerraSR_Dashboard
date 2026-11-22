"""Streamlit-friendly adapter around the project's ORIGINAL predict.py.

The adapter deliberately keeps checkpoint loading, Sentinel-2 normalization,
model inference and GeoTIFF export consistent with the supplied project.
"""
from __future__ import annotations

import io
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluate import load_model
from predict import process, sr_tiled, to_unit


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npz"}
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "runs" / "exp1" / "best.pt"


@dataclass
class PredictionResult:
    input_image: Image.Image
    bicubic_image: Image.Image
    sr_image: Image.Image
    native_bytes: bytes
    native_extension: str
    preview_png: bytes
    source_size: tuple[int, int]
    result_size: tuple[int, int]
    seconds: float
    reference_image: Image.Image | None = None
    metrics: dict[str, float] | None = None
    geographic: bool = False


def choose_device(requested: str) -> torch.device:
    if requested == "CUDA":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was selected but no CUDA-enabled PyTorch GPU is available.")
        return torch.device("cuda")
    if requested == "CPU":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(checkpoint: str, requested_device: str = "Auto"):
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    device = choose_device(requested_device)
    # Load only checkpoints created by your own train.py, not untrusted files.
    model, scale = load_model(str(path), device)
    if scale < 1:
        raise ValueError(f"Invalid scale in checkpoint: {scale}")
    model.eval()
    return model, int(scale), device


def as_image(chw: np.ndarray) -> Image.Image:
    """Convert a [0, 1] RGB CHW array to an 8-bit PIL preview."""
    chw = np.clip(chw, 0.0, 1.0)
    return Image.fromarray(np.rint(chw.transpose(1, 2, 0) * 255.0).astype(np.uint8), "RGB")


def image_png_bytes(img: Image.Image) -> bytes:
    stream = io.BytesIO()
    img.save(stream, format="PNG")
    return stream.getvalue()


def check_output_size(width: int, height: int, scale: int, max_output_mp: int) -> None:
    if width < 1 or height < 1:
        raise ValueError("Image has invalid width or height.")
    out_pixels = width * height * scale * scale
    if out_pixels > max_output_mp * 1_000_000:
        raise ValueError(
            f"Result would be {width * scale} x {height * scale} "
            f"({out_pixels / 1_000_000:.1f} megapixels), beyond your "
            f"{max_output_mp} MP limit. Crop the image or increase the limit in the sidebar."
        )


def bicubic_unit(chw: np.ndarray, scale: int) -> np.ndarray:
    with torch.no_grad():
        ten = torch.from_numpy(np.ascontiguousarray(chw))[None]
        return F.interpolate(ten, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)[0].numpy()


def read_uploaded_image(path: Path, mode: str, refl_max: float,
                        offset: float | None, bands: tuple[int, int, int]):
    """Match predict.process input interpretation for GUI previews."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        with rasterio.open(path) as src:
            if max(bands) > src.count:
                raise ValueError(f"GeoTIFF has {src.count} bands, but RGB indices are {bands}.")
            width, height = src.width, src.height
            raw = src.read(list(bands)).astype(np.float32)
            dtype = src.dtypes[0]
            tags = src.tags()
        used_mode = mode if mode != "auto" else (
            "uint8" if dtype == "uint8" else
            "s2" if dtype.startswith(("uint", "int")) else "float"
        )
        used_offset = offset if offset is not None else float(tags.get("S2_REFLECTANCE_OFFSET", 0.0))
        return to_unit(raw, used_mode, refl_max, used_offset).astype(np.float32), (width, height)

    with Image.open(path) as im:
        width, height = im.size
        raw = np.asarray(im.convert("RGB")).transpose(2, 0, 1).astype(np.float32)
    used_mode = mode if mode != "auto" else "uint8"
    used_offset = offset if offset is not None else 0.0
    return to_unit(raw, used_mode, refl_max, used_offset).astype(np.float32), (width, height)


def compute_reference_metrics(bic: np.ndarray, pred: np.ndarray, hr: np.ndarray):
    from train import psnr_batch, ssim_batch
    if hr.shape != pred.shape or min(hr.shape[1:]) < 11:
        return None
    with torch.no_grad():
        ref = torch.from_numpy(hr.copy())[None]
        bic_t = torch.from_numpy(bic.copy())[None]
        pred_t = torch.from_numpy(pred.copy())[None]
        return {
            "psnr_bicubic": float(psnr_batch(bic_t, ref)[0]),
            "psnr_cnn": float(psnr_batch(pred_t, ref)[0]),
            "ssim_bicubic": float(ssim_batch(bic_t, ref)[0]),
            "ssim_cnn": float(ssim_batch(pred_t, ref)[0]),
        }


def process_upload(
    data: bytes,
    filename: str,
    model: torch.nn.Module,
    scale: int,
    device: torch.device,
    *,
    mode: str = "auto",
    bands: tuple[int, int, int] = (1, 2, 3),
    offset: float | None = None,
    refl_max: float = 0.3,
    tile: int = 128,
    overlap: int = 16,
    max_output_mp: int = 8,
) -> PredictionResult:
    """Return actual CNN inference results and download-ready bytes.

    - PNG/JPEG -> PNG result
    - GeoTIFF -> georeferenced GeoTIFF result and PNG preview
    - cleaned dataset .npz (lr/optional hr) -> PNG + optional PSNR/SSIM
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext or '(none)'}")
    if not data:
        raise ValueError("The uploaded file is empty.")
    if refl_max <= 0:
        raise ValueError("Reflectance maximum must be greater than zero.")
    if tile <= 2 * overlap or overlap < 0:
        raise ValueError("Tile size must be greater than twice the overlap.")
    if len(bands) != 3 or min(bands) < 1 or len(set(bands)) != 3:
        raise ValueError("Enter three different, positive 1-based RGB band indices, e.g. 1,2,3.")

    started = time.perf_counter()
    if ext == ".npz":
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            if "lr" not in z:
                raise ValueError("NPZ must contain an 'lr' array (3,H,W).")
            lr = z["lr"].astype(np.float32)
            hr = z["hr"].astype(np.float32) if "hr" in z else None
        if lr.ndim != 3 or lr.shape[0] != 3 or not np.isfinite(lr).all():
            raise ValueError("NPZ 'lr' must be a finite RGB array with shape (3,H,W).")
        height, width = lr.shape[1:]
        check_output_size(width, height, scale, max_output_mp)
        lr = np.clip(lr, 0, 1)
        bic = bicubic_unit(lr, scale)
        sr = sr_tiled(model, torch.from_numpy(np.ascontiguousarray(lr)), scale, device,
                      tile=tile, overlap=overlap).numpy()
        output_image = as_image(sr)
        ref_image, metrics = None, None
        if hr is not None and hr.shape == sr.shape and np.isfinite(hr).all():
            hr = np.clip(hr, 0, 1)
            ref_image = as_image(hr)
            metrics = compute_reference_metrics(bic, sr, hr)
        png_bytes = image_png_bytes(output_image)
        return PredictionResult(as_image(lr), as_image(bic), output_image,
                                png_bytes, ".png", png_bytes, (width, height),
                                (width * scale, height * scale), time.perf_counter() - started,
                                reference_image=ref_image, metrics=metrics)

    with tempfile.TemporaryDirectory(prefix="satellite_sr_") as tmp:
        source = Path(tmp) / ("input" + ext)  # user filename never becomes a filesystem path
        source.write_bytes(data)
        # Validate the output budget from file headers BEFORE decoding raster data.
        if ext in {".tif", ".tiff"}:
            import rasterio
            with rasterio.open(source) as header:
                check_output_size(header.width, header.height, scale, max_output_mp)
        else:
            with Image.open(source) as header:
                check_output_size(header.width, header.height, scale, max_output_mp)
        lr, (width, height) = read_uploaded_image(source, mode, refl_max, offset, bands)
        bic = bicubic_unit(lr, scale)
        geographic = ext in {".tif", ".tiff"}
        target_ext = ".tif" if geographic else ".png"
        destination = Path(tmp) / ("enhanced" + target_ext)
        args = SimpleNamespace(mode=mode, bands=list(bands), offset=offset,
                               refl_max=refl_max, tile=tile, overlap=overlap,
                               compare=False)
        # Use supplied prediction pipeline for inference and georeferenced export.
        process(source, destination, model, scale, device, args)
        native = destination.read_bytes()
        preview = destination.with_name(destination.stem + "_preview.png").read_bytes()
        with Image.open(io.BytesIO(preview)) as im:
            output_image = im.convert("RGB").copy()
        return PredictionResult(as_image(lr), as_image(bic), output_image,
                                native, target_ext, preview, (width, height),
                                (width * scale, height * scale), time.perf_counter() - started,
                                geographic=geographic)


def limit_preview(image: Image.Image, max_side: int = 1250) -> Image.Image:
    """Shrink display-only copies, keeping original downloadable pixels untouched."""
    if max(image.size) <= max_side:
        return image
    preview = image.copy()
    preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return preview


def wipe_comparison(left: Image.Image, right: Image.Image, position_pct: int) -> Image.Image:
    """Pixel-registered swipe: left half from left; remainder from right."""
    from PIL import ImageDraw
    width, height = right.size
    if left.size != right.size:
        left = left.resize(right.size, Image.Resampling.NEAREST)
    pos = max(0, min(width, round(width * position_pct / 100)))
    merged = right.copy()
    if pos:
        merged.paste(left.crop((0, 0, pos, height)), (0, 0))
    draw = ImageDraw.Draw(merged)
    draw.line([(pos, 0), (pos, height)], fill=(242, 198, 94), width=3)
    return merged
