"""Integration smoke tests: use an untrained checkpoint ONLY to verify I/O wiring.
These do NOT demonstrate scientific image quality. Replace with your real best.pt.
"""
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from model import build_model
from sr_backend import load_trained_model, process_upload, wipe_comparison


@pytest.fixture(scope="module")
def loaded_model(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("weights")
    checkpoint = tmp / "dummy_untrained.pt"
    model = build_model("srcnn", scale=2)
    torch.save({"model": model.state_dict(), "args": {"model": "srcnn", "scale": 2}}, checkpoint)
    return load_trained_model(str(checkpoint), "CPU")


def test_png_upload(loaded_model):
    model, scale, device = loaded_model
    rgb = np.random.default_rng(0).integers(0, 255, (35, 44, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    result = process_upload(stream.getvalue(), "safe.png", model, scale, device)
    assert result.source_size == (44, 35)
    assert result.result_size == (88, 70)
    assert result.native_extension == ".png"
    assert Image.open(io.BytesIO(result.native_bytes)).size == (88, 70)


def test_geotiff_preserves_footprint(loaded_model):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    model, scale, device = loaded_model
    raw = np.full((3, 22, 30), 2100, dtype=np.uint16)
    transform = from_origin(499000, 1290000, 10, 10)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=30, height=22, count=3,
                      dtype="uint16", crs="EPSG:32643", transform=transform) as dst:
            dst.write(raw)
            dst.update_tags(S2_REFLECTANCE_OFFSET="-0.1")
        blob = mem.read()
    result = process_upload(blob, "scene.tif", model, scale, device)
    assert result.geographic and result.native_extension == ".tif"
    with MemoryFile(result.native_bytes) as mem:
        with mem.open() as dst:
            assert (dst.width, dst.height) == (60, 44)
            assert dst.crs.to_epsg() == 32643
            assert dst.transform.a == pytest.approx(5)
            assert dst.transform.e == pytest.approx(-5)
            assert dst.bounds.left == pytest.approx(499000)
            assert dst.bounds.top == pytest.approx(1290000)


def test_npz_with_reference_metrics(loaded_model):
    model, scale, device = loaded_model
    rng = np.random.default_rng(42)
    lr = rng.random((3, 24, 24), dtype=np.float32)
    hr = rng.random((3, 48, 48), dtype=np.float32)
    stream = io.BytesIO()
    np.savez(stream, lr=lr, hr=hr)
    result = process_upload(stream.getvalue(), "test.npz", model, scale, device)
    assert result.reference_image is not None
    assert result.metrics is not None
    assert result.metrics["psnr_cnn"] > 0


def test_large_upload_guard(loaded_model):
    model, scale, device = loaded_model
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    with pytest.raises(ValueError, match="beyond your"):
        process_upload(stream.getvalue(), "img.png", model, scale, device,
                       max_output_mp=0)


def test_wipe():
    left = Image.new("RGB", (40, 30), (255, 0, 0))
    right = Image.new("RGB", (40, 30), (0, 0, 255))
    result = wipe_comparison(left, right, 50)
    assert result.getpixel((0, 5)) == (255, 0, 0)
    assert result.getpixel((39, 5)) == (0, 0, 255)


def test_checkpoint_drives_x3_scale(tmp_path):
    # The UI must use scale in the checkpoint, not a hardcoded x2 assumption.
    checkpoint = tmp_path / "demo_x3.pt"
    model = build_model("resnet", scale=3, feats=8, n_blocks=1)
    torch.save({"model": model.state_dict(),
                "args": {"model": "resnet", "scale": 3, "feats": 8, "blocks": 1}}, checkpoint)
    loaded, scale, device = load_trained_model(str(checkpoint), "CPU")
    assert scale == 3
    img = Image.new("RGB", (20, 18), (120, 110, 90))
    stream = io.BytesIO()
    img.save(stream, format="PNG")
    result = process_upload(stream.getvalue(), "patch.png", loaded, scale, device)
    assert result.result_size == (60, 54)
