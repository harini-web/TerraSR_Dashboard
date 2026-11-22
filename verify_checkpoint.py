"""Check that your own trained DeBlur checkpoint can load and produce a prediction.

Run from DeBlur_Dashboard:
    python verify_checkpoint.py
or:
    python verify_checkpoint.py "D:\\path\\to\\best.pt"

Never load checkpoints downloaded from untrusted sources.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    folder = Path(__file__).resolve().parent
    path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else folder / 'runs' / 'exp1' / 'best.pt'
    if not path.is_absolute():
        path = (folder / path).resolve()
    print(f'Checkpoint: {path}')
    if not path.is_file():
        print('\nNOT FOUND: Copy the trained best.pt here, or pass its full Windows path.')
        print('This ZIP includes model.py (architecture) but cannot include your learned weights.')
        return 1

    try:
        import torch
        from evaluate import load_model
        model, scale = load_model(str(path), torch.device('cpu'))
        model.eval()
        with torch.inference_mode():
            sample = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
            output = model(sample)
        expected = (1, 3, 16 * int(scale), 16 * int(scale))
        if tuple(output.shape) != expected:
            print(f'Unexpected output: {tuple(output.shape)}; expected {expected}')
            return 2
        parameters = sum(p.numel() for p in model.parameters())
        print(f'PASS: Trained checkpoint loads successfully.')
        print(f'Upscale factor: {scale}x')
        print(f'Model parameters: {parameters:,}')
        print(f'Test input (RGB): 16 x 16 pixels')
        print(f'Test output (RGB): {output.shape[3]} x {output.shape[2]} pixels')
        print('You can now run: python app.py')
        return 0
    except Exception as exc:
        print(f'FAILED TO LOAD: {type(exc).__name__}: {exc}')
        print('Make sure best.pt was saved by this project\'s train.py and matches model.py.')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
