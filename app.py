"""DeBlur Dashboard — local web interface for Sentinel-2 super-resolution.

Start with: python app.py
Open: http://127.0.0.1:5000

This server accepts images from the same computer only (127.0.0.1). It loads
only your own, trusted PyTorch checkpoint; an example checkpoint is not included.
"""
from __future__ import annotations

import base64
import io
import os
import re
import secrets
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from sr_backend import (
    DEFAULT_CHECKPOINT,
    image_png_bytes,
    limit_preview,
    load_trained_model,
    process_upload,
)

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 80 * 1024 * 1024  # 80MB. This is a local, single-machine application.
MAX_RESULTS_PER_SESSION = 4
app = FastAPI(title="DeBlur", docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32), same_site="lax", https_only=False)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

_model_cache: dict[tuple[str, int, str], tuple] = {}
_model_lock = threading.RLock()
_results: dict[str, OrderedDict[str, dict]] = {}
_results_lock = threading.RLock()


def visitor_id(request: Request) -> str:
    if "visitor_id" not in request.session:
        request.session["visitor_id"] = uuid.uuid4().hex
    return request.session["visitor_id"]


def checkpoint_path(text: str | None) -> Path:
    raw = (text or "runs/exp1/best.pt").strip().strip('"')
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def get_model(path: Path, processor: str):
    if not path.is_file():
        raise FileNotFoundError(
            "Your trained best.pt has not been found. Copy it to runs/exp1/best.pt, "
            "or set its full local path in Settings. The Python scripts alone do not contain learned weights."
        )
    key = (str(path), path.stat().st_mtime_ns, processor)
    with _model_lock:
        if key not in _model_cache:
            _model_cache.clear()
            _model_cache[key] = load_trained_model(str(path), processor)
        return _model_cache[key]


def preview_data_uri(image) -> str:
    png = image_png_bytes(limit_preview(image, 1000))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def safe_stem(filename: str) -> str:
    basename = Path(filename.replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(basename).stem)[:64]
    return stem or "satellite_image"


def parse_settings(
    mode: str, processor: str, bands: str, offset: str,
    refl_max: str, tile: str, overlap: str, max_output_mp: str,
) -> dict:
    if mode not in {"auto", "s2", "uint8", "float"}:
        raise ValueError("Input mode must be Auto, Sentinel-2 DN, 8-bit RGB, or Float.")
    if processor not in {"Auto", "CUDA", "CPU"}:
        raise ValueError("Processor must be Auto, CUDA or CPU.")
    try:
        chosen_bands = tuple(int(v.strip()) for v in bands.split(","))
        max_refl = float(refl_max)
        tile_size = int(tile)
        border = int(overlap)
        budget = int(max_output_mp)
        dn_offset = float(offset) if offset.strip() else None
    except ValueError as exc:
        raise ValueError("A numeric setting is invalid. Open Settings and review the values.") from exc
    if len(chosen_bands) != 3 or min(chosen_bands) < 1 or len(set(chosen_bands)) != 3:
        raise ValueError("RGB bands must be three different positive numbers (e.g. 1,2,3).")
    if not (0 < max_refl <= 2.0):
        raise ValueError("Reflectance stretch must be greater than 0 and at most 2.")
    if tile_size not in {64, 96, 128, 160, 192, 256} or not 0 <= border <= 32 or tile_size <= 2 * border:
        raise ValueError("Tile / overlap configuration is invalid. Use tile 128 and overlap 16.")
    if not 1 <= budget <= 30:
        raise ValueError("Output size limit must be between 1 and 30 megapixels.")
    return dict(mode=mode, processor=processor, bands=chosen_bands, offset=dn_offset,
                refl_max=max_refl, tile=tile_size, overlap=border, max_output_mp=budget)


def summary(token: str, rec: dict) -> dict:
    return {
        "id": token,
        "name": rec["name"],
        "created_at": rec["created_at"],
        "input_size": rec["input_size"],
        "output_size": rec["output_size"],
        "scale": rec["scale"],
        "duration": rec["duration"],
        "geographic": rec["geographic"],
        "metrics": rec["metrics"],
        "processor": rec["processor"],
        "download_name": rec["download_name"],
        "download_url": f"/api/download/{token}/enhanced",
        "preview_download_url": f"/api/download/{token}/preview",
    }


@app.get("/")
def index(request: Request):
    visitor_id(request)
    return FileResponse(ROOT / "templates" / "index.html")


@app.get("/api/status")
def status(request: Request, checkpoint: str = "runs/exp1/best.pt"):
    owner = visitor_id(request)
    path = checkpoint_path(checkpoint)
    with _results_lock:
        completed = len(_results.get(owner, {}))
    return {
        "checkpoint_present": path.is_file(),
        "checkpoint_name": path.name,
        "default_checkpoint": str(DEFAULT_CHECKPOINT),
        "gpu_available": torch.cuda.is_available(),
        "current_count": completed,
    }


@app.post("/api/enhance")
def enhance(
    request: Request,
    image: UploadFile = File(...),
    checkpoint: str = Form("runs/exp1/best.pt"),
    processor: str = Form("Auto"),
    mode: str = Form("auto"),
    bands: str = Form("1,2,3"),
    offset: str = Form(""),
    refl_max: str = Form("0.3"),
    tile: str = Form("128"),
    overlap: str = Form("16"),
    max_output_mp: str = Form("8"),
):
    owner = visitor_id(request)
    filename = Path((image.filename or "").replace("\\", "/")).name
    if not filename:
        raise HTTPException(400, "Choose a low-resolution image first.")
    try:
        opts = parse_settings(mode, processor, bands, offset, refl_max, tile, overlap, max_output_mp)
        incoming = image.file.read(MAX_UPLOAD + 1)
        if len(incoming) > MAX_UPLOAD:
            raise HTTPException(413, "Image exceeds the 80 MB upload limit. Crop it and try again.")
        model, scale, device = get_model(checkpoint_path(checkpoint), opts["processor"])
        with _model_lock:
            output = process_upload(
                incoming, filename, model, scale, device,
                mode=opts["mode"], bands=opts["bands"], offset=opts["offset"],
                refl_max=opts["refl_max"], tile=opts["tile"], overlap=opts["overlap"],
                max_output_mp=opts["max_output_mp"],
            )
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError, OSError, KeyError, RuntimeError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, "Unexpected inference error. See the terminal for more details.") from exc

    token = uuid.uuid4().hex
    record = {
        "name": filename,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_size": list(output.source_size),
        "output_size": list(output.result_size),
        "scale": scale,
        "duration": round(output.seconds, 2),
        "geographic": output.geographic,
        "metrics": output.metrics,
        "processor": str(device).upper(),
        "download_name": f"{safe_stem(filename)}_CNN_x{scale}{output.native_extension}",
        "native_bytes": output.native_bytes,
        "preview_bytes": output.preview_png,
        "previews": {
            "input": preview_data_uri(output.input_image),
            "bicubic": preview_data_uri(output.bicubic_image),
            "enhanced": preview_data_uri(output.sr_image),
            "reference": preview_data_uri(output.reference_image) if output.reference_image else None,
        },
    }
    with _results_lock:
        history = _results.setdefault(owner, OrderedDict())
        history[token] = record
        history.move_to_end(token)
        while len(history) > MAX_RESULTS_PER_SESSION:
            history.popitem(last=False)
    return {**summary(token, record), "previews": record["previews"]}


@app.get("/api/history")
def history(request: Request):
    owner = visitor_id(request)
    with _results_lock:
        return {"items": [summary(token, rec) for token, rec in reversed(_results.get(owner, {}).items())]}


@app.get("/api/result/{token}")
def result_info(request: Request, token: str):
    owner = visitor_id(request)
    with _results_lock:
        rec = _results.get(owner, {}).get(token)
        if not rec:
            raise HTTPException(404, "Result not found (results clear when the server restarts).")
        return {**summary(token, rec), "previews": rec["previews"]}


@app.get("/api/download/{token}/{kind}")
def download(request: Request, token: str, kind: str):
    owner = visitor_id(request)
    with _results_lock:
        rec = _results.get(owner, {}).get(token)
        if not rec:
            raise HTTPException(404, "Result has expired. Enhance the image again.")
        if kind == "enhanced":
            payload, name = rec["native_bytes"], rec["download_name"]
            mime = "image/tiff" if rec["geographic"] else "image/png"
        elif kind == "preview":
            payload, name = rec["preview_bytes"], Path(rec["download_name"]).stem + "_preview.png"
            mime = "image/png"
        else:
            raise HTTPException(404, "Unknown download type.")
    return Response(payload, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.exception_handler(HTTPException)
def http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


if __name__ == "__main__":
    import uvicorn

    print("\nDeBlur Dashboard: http://127.0.0.1:5000")
    print("Use your own trusted runs/exp1/best.pt checkpoint.\n")
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("DeBlur_PORT", "5000")), log_level="warning")
