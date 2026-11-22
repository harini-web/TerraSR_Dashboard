"""API smoke tests with temporary untrained weights. Never represent image quality."""
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import app
from model import build_model


def make_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / 'test_random_weights.pt'
    model = build_model('srcnn', scale=2)
    torch.save({'model': model.state_dict(), 'args': {'model': 'srcnn', 'scale': 2}}, checkpoint)
    return checkpoint


def make_image_bytes():
    im = Image.fromarray(np.random.default_rng(3).integers(0, 256, (27, 33, 3), dtype=np.uint8))
    f = BytesIO()
    im.save(f, format='PNG')
    return f.getvalue()


def test_dashboard_html_and_static_files():
    with TestClient(app) as client:
        resp = client.get('/')
        assert resp.status_code == 200
        assert 'Upload a low-resolution image' in resp.text
        assert 'Enhance image now' in resp.text
        assert 'Download enhanced image' in resp.text
        assert client.get('/static/styles.css').status_code == 200
        assert client.get('/static/app.js').status_code == 200


def test_checkpoint_missing_is_clear():
    with TestClient(app) as client:
        r = client.get('/api/status', params={'checkpoint': 'definitely-not-present.pt'})
        assert r.status_code == 200 and not r.json()['checkpoint_present']
        r = client.post('/api/enhance', data={'checkpoint':'definitely-not-present.pt'},
                        files={'image': ('scene.png', make_image_bytes(), 'image/png')})
        assert r.status_code == 400
        assert 'best.pt' in r.json()['error']


def test_upload_inference_history_and_download(tmp_path):
    ckpt = make_checkpoint(tmp_path)
    with TestClient(app) as client:
        state = client.get('/api/status', params={'checkpoint': str(ckpt)}).json()
        assert state['checkpoint_present']
        resp = client.post('/api/enhance', data={'checkpoint':str(ckpt), 'processor':'CPU'},
                           files={'image': ('scene.png', make_image_bytes(), 'image/png')})
        assert resp.status_code == 200, resp.text
        obj = resp.json()
        assert obj['scale'] == 2
        assert obj['input_size'] == [33,27]
        assert obj['output_size'] == [66,54]
        assert obj['previews']['enhanced'].startswith('data:image/png;base64,')
        assert obj['previews']['input'].startswith('data:image/png;base64,')
        assert obj['download_url'].startswith('/api/download/')
        dl = client.get(obj['download_url'])
        assert dl.status_code == 200
        assert Image.open(BytesIO(dl.content)).size == (66,54)
        hist = client.get('/api/history').json()['items']
        assert hist[0]['name'] == 'scene.png'
        assert client.get('/api/result/' + obj['id']).status_code == 200
        assert client.get('/api/download/' + obj['id'] + '/bad').status_code == 404
        with TestClient(app) as other_browser:
            assert other_browser.get(obj['download_url']).status_code == 404


def test_rejects_invalid_settings(tmp_path):
    ckpt = make_checkpoint(tmp_path)
    with TestClient(app) as client:
        resp = client.post('/api/enhance', data={'checkpoint':str(ckpt), 'bands':'1,1,1'},
                           files={'image': ('scene.png', make_image_bytes(), 'image/png')})
        assert resp.status_code == 400
        assert 'RGB bands' in resp.json()['error']
