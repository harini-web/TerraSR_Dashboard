# START HERE — DeBlur, your complete Windows setup guide

This guide is for **DeBlur_User_Friendly_Dashboard.zip**, the FastAPI/HTML/CSS/JS dashboard.
It is **not** the older Streamlit app. The dashboard is a real UI for your existing
Sentinel-2 super-resolution PyTorch CNN; it cannot perform real inference until
it receives your own trained checkpoint (`best.pt`).

## A. Which files go where?

After extracting the ZIP, your folder must look like this:

```text
DeBlur_Dashboard\
    app.py                        <- local Python web server
    sr_backend.py                 <- connects web upload to CNN prediction
    model.py                      <- CNN ARCHITECTURE (already included)
    evaluate.py                   <- loads your weights + model settings
    predict.py                    <- performs super-resolution + TIFF output
    train.py                      <- optional: train a new CNN
    verify_checkpoint.py          <- check your learned weights first
    requirements-dashboard.txt
    start_windows.bat
    templates\index.html          <- dashboard page
    static\styles.css             <- appearance
    static\app.js                 <- buttons, file upload and results
    runs\
        exp1\
            best.pt               <- YOU MUST COPY THIS TRAINED CHECKPOINT
```

**The trained model has two parts:** `model.py` describes the network, while
`best.pt` contains its learned weights and training settings. `model.py` is
already included, but the original uploads did not contain `best.pt`.

Do **not** rename `best.pt`, replace `model.py` with an unrelated network, or
move your image dataset into the web directory just to run inference.

## B. Find best.pt on your computer

If you trained using the supplied `train.py`, look in your original project at:

```text
<original-training-project>\runs\exp1\best.pt
```

Or open Windows File Explorer and search for `best.pt` in your training folder.
Copy (not move) the file to:

```text
<extracted-app>\DeBlur_Dashboard\runs\exp1\best.pt
```

If you trained with a different `--out` folder, the location is `<out>\best.pt`.
You can keep the checkpoint in its old location and instead put its full path in
the dashboard under **Model settings > Trained checkpoint path**.

If you have only `model.py` but **no trained best.pt**, you need to train the
network before the app can enhance images; see section H below.

## C. Open Windows CMD in the correct folder

1. Extract the ZIP.
2. In File Explorer, open **DeBlur_Dashboard** (the folder containing `app.py`).
3. Click the File Explorer address bar, type **cmd**, then press Enter.
4. The prompt should now end with `\DeBlur_Dashboard>`.

This avoids mistakes caused by guessing the full path.

## D. Install Python and create your environment (once)

In that **CMD** window (not PowerShell), type:

```cmd
python --version
py --list
```

Use **64-bit Python 3.11 or 3.12** for this guide. If Python is missing,
install it from https://www.python.org/downloads/windows/ and open a NEW CMD.
Do not run `py -3.11` unless `py --list` actually shows Python 3.11.

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-dashboard.txt
```

When CMD is activated, your prompt starts with `(.venv)`.
CUDA is optional; CPU inference works without an NVIDIA GPU.
For a CUDA installation, use PyTorch's official install instructions
(https://pytorch.org/get-started/locally/) in this same `.venv`.

## E. Verify your own checkpoint (before opening the app)

In the activated CMD, from `DeBlur_Dashboard`, run:

```cmd
dir runs\exp1\best.pt
python verify_checkpoint.py
```

A good result says `PASS: Trained checkpoint loads successfully` and shows
its actual scale (for example, `2x`). This check uses a tiny dummy image ONLY
to test model loading and the output dimensions; **it does not measure image
quality**.

If your checkpoint is outside the app folder:

```cmd
python verify_checkpoint.py "D:\YourProject\runs\exp1\best.pt"
```

Only load your own, trusted checkpoint files.

## F. Run the dashboard

```cmd
python app.py
```

Keep CMD open while using the application, and open:

**http://127.0.0.1:5000**

Or double-click `start_windows.bat` once everything is installed. The batch
script uses `.venv` automatically if it exists.

Visit `http://127.0.0.1:5000/api/status` if you want a quick configuration
check. With the default checkpoint path, `"checkpoint_present": true` means
it has found the file. `verify_checkpoint.py` is the stronger test because it
actually loads the weights.

## G. Upload a low-resolution image -> enhance -> download

1. In the left menu, select **Enhance image**.
2. Find **01 — Upload a low-resolution image**. Drag an image into the dotted
   box or click **Browse files**. You can upload `.tif/.tiff`, `.png`, `.jpg`,
   `.jpeg`, or a cleaned project `.npz` test pair.
3. Find **02 — Enhance with your CNN** and click **Enhance image now**.
4. The app displays original low-resolution image, bicubic baseline, and CNN
   super-resolution. Drag the before/after slider to compare.
5. Click **Download enhanced image**. GeoTIFF input produces a georeferenced
   TIFF; PNG/JPEG input produces a PNG. You can also download a PNG preview.

For best results, use a **3-band Sentinel-2 Level-2A RGB GeoTIFF** matching
training normalization (red, green, blue bands). PNG/JPEG can be demonstrated
but may not match the sensor representation used for training.

You do **not** browse for the model at the image upload box. The app reads the
checkpoint from its path; the upload box is only for the **low-resolution image**.

## H. No best.pt yet? Train it first

Training requires your own cleaned `.npz` pairs and `data\clean\splits.json`.
Copy your **data\clean** folder from the training project here (or train in
the original folder and copy its checkpoint afterward). Install the full
training dependencies:

```cmd
python -m pip install -r requirements.txt
python train.py --data data/clean --out runs/exp1 --epochs 300 --workers 0
```

If using a supported CUDA PyTorch install, you may add `--amp` to training.
The training script creates `runs\exp1\best.pt` when validation improves.

**Important:** With original defaults, `--scale 2` trains 2x (10 m -> nominal
5 m). A 3x or 4x network requires appropriate matching HR training pairs, a
compatible `--scale`, and new training. Changing a GUI option cannot create a
model at a scale it was not trained for.

If you need to obtain NEW Sentinel-2 samples using the optional
`fetch_s2_rgb.py` or `download_data.py`, see `README_original.md`. Those scripts
also require network access to the data providers. You do not need those
services once you have an input image and `best.pt`.

## I. What links the web app to your CNN?

The pieces are **already linked in the supplied code**. You do not manually
copy model logic into HTML or edit Python import statements.

```text
Browser: templates/index.html + static/app.js
       -> HTTP POST /api/enhance in app.py
       -> sr_backend.py: load_trained_model()
       -> evaluate.py: load_model(best.pt, device)
       -> model.py: build_model() + learned weights
       -> sr_backend.py: process_upload()
       -> predict.py: process() / sr_tiled()
       -> response: RGB preview + PNG/GeoTIFF download
```

`static/styles.css` only controls how the page looks. `train.py` is needed
for training, not for the normal upload -> inference workflow (its helpers
also provide metrics for the optional `.npz` ground-truth comparison).

## J. Common problems

| Problem                             | What to do                                                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `python` not recognized             | Install 64-bit Python 3.11/3.12, reopen CMD, run `python --version`.                                             |
| `py -3.11` runtime not found        | Use `python -m venv .venv` with an installed supported version.                                                  |
| `Activate.ps1` path not found       | In CMD use `.venv\Scripts\activate.bat`, only _after_ venv creation.                                             |
| `No module named fastapi` / `torch` | Activate `.venv`; run `python -m pip install -r requirements-dashboard.txt`.                                     |
| `Model not found`                   | Check `dir runs\exp1\best.pt`; fix the path in **Model settings** if elsewhere.                                  |
| Error reading `best.pt`             | Ensure checkpoint came from this project's `train.py` and matches `model.py`; run `python verify_checkpoint.py`. |
| `CUDA` unavailable                  | Select **Auto** or **CPU** in Model settings.                                                                    |
| App does not open                   | Keep `python app.py` CMD running; visit `http://127.0.0.1:5000`.                                                 |
| Port 5000 already used              | In CMD `set DeBlur_PORT=5001` then `python app.py`; open `http://127.0.0.1:5001`.                                |
| Output dark / colors odd            | Check input interpretation, RGB band order, reflectance max and offset.                                          |
| Image too large                     | Crop it, or increase **Maximum output size** carefully (RAM usage grows).                                        |
| No PSNR/SSIM                        | Only possible with an aligned HR reference, e.g. a cleaned `.npz` pair.                                          |

For a quick package smoke test after installation:

```cmd
python -m pip install pytest httpx
python -m pytest -q tests
```

Those tests validate the application plumbing using temporary **untrained**
weights. Only your trained `best.pt` supplies meaningful super-resolution.
