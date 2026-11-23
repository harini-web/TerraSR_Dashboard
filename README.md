# DeBlur Dashboard — User-friendly Satellite Image Enhancer

**A redesigned browser interface for the trained Sentinel-2 super-resolution CNN.**

The light dashboard uses a left sidebar, pastel status cards, a prominent drag-and-drop upload area, a single purple enhancement button, three-way visual comparisons, a live before/after slider and clear download buttons. Inspired by the dashboard layout you supplied, but built specifically for satellite image processing.

**Important:** This is an actual Python-powered application, not a simulated visual demo. It calls your original `predict.py` and loads your checkpoint using `evaluate.py` and `model.py`. No model weights were uploaded, so you must add your own trained `runs/exp1/best.pt`. The app never substitutes random demo weights for your trained model.

## Step-by-step: Windows CMD

1. Extract `DeBlur_User_Friendly_Dashboard.zip` anywhere, then open the folder named `DeBlur_Dashboard` that contains `app.py`.
2. In that folder's Windows Command Prompt, check your Python version:

   ```cmd
   python --version
   py --list
   ```

   Python 3.11 or 3.12, 64-bit, is recommended. If `py -3.11` fails but `python --version` reports 3.11 or 3.12, use `python` instead of `py -3.11`.

3. Create the virtual environment **in the app folder**:

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

   Use `activate.bat` in CMD, **not** `Activate.ps1` (that is for PowerShell). After activation the prompt starts with `(.venv)`.

4. Install the dashboard and inference dependencies (smaller than the full training requirements):

   ```cmd
   python -m pip install --upgrade pip
   python -m pip install -r requirements-dashboard.txt
   ```

   CUDA acceleration is optional. CPU processing also works. For an NVIDIA GPU, you may need to install the matching CUDA PyTorch version according to the official PyTorch instructions.

5. **Add your trained checkpoint.** Copy the actual `best.pt` created by your training script:

   ```text
   DeBlur_Dashboard/
   ├─ app.py
   ├─ model.py
   ├─ predict.py
   └─ runs/
      └─ exp1/
         └─ best.pt   <-- YOU NEED TO PROVIDE THIS
   ```

   You can alternatively enter the full checkpoint path under **Model settings**.

6. Run the application:

   ```cmd
   python app.py
   ```

   Open **http://127.0.0.1:5000** in your browser.

   You can also double-click `start_windows.bat` after setup. It automatically uses `.venv\Scripts\python.exe` when available.

## Where to upload the low-resolution image

1. Click **Enhance image** in the left sidebar (this is also the first screen when you open the app).
2. In the card named **01 — Upload a low-resolution image**, drag your image into the dotted box, or click **Browse files**.
3. The file name and preview (for PNG/JPEG) appear just below the box.
4. In the card named **02 — Enhance with your CNN**, click the large purple **Enhance image now** button.
5. On completion the **03 — Your enhanced image is ready** section appears: **Original LR**, **Bicubic baseline**, **CNN super-resolution**. Move the comparison slider to inspect differences.
6. Click **Download enhanced PNG** or **Download GeoTIFF**, depending on your input. A PNG preview is also offered.

The output **scale is read from `best.pt`**. With the original training defaults, an x2 model will map 10 m input pixel spacing to nominal 5 m output pixel spacing. This is not a verified real new sensor resolution.

## Explanation of every menu option

| Left menu          | Meaning                                                                                                               |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **Dashboard**      | Model availability, number of images processed in this browser session, most recent scale and a workflow explanation. |
| **Enhance image**  | The important page: choose the low-resolution file, run super-resolution, compare and download.                       |
| **Recent results** | The last four processed images from this browser while the Python app remains running. Open or download them.         |
| **How to use**     | Practical steps, supported input files and honest limits of the project.                                              |
| **Model settings** | Checkpoint path, GPU/CPU choice, band order, preprocessing and memory settings. Keep defaults unless necessary.       |

### Explanation of advanced model settings

| Setting                         | Meaning                                                                                      | Recommended default |
| ------------------------------- | -------------------------------------------------------------------------------------------- | ------------------- |
| **Trained checkpoint path**     | Location of model weights produced by `train.py`. **Required** for prediction.               | `runs/exp1/best.pt` |
| **Processing device**           | Auto uses NVIDIA CUDA if available, else CPU.                                                | Auto                |
| **Input interpretation**        | PNG/JPEG usually use 8-bit normalization; 16-bit S2 TIFF uses reflectance DN normalization.  | Auto                |
| **RGB band indices**            | Selects GeoTIFF red, green and blue bands (1-based).                                         | `1,2,3`             |
| **Reflectance stretch maximum** | Sentinel-2 physical reflectance divided by this maximum (must match training).               | `0.3`               |
| **Reflectance offset**          | Accounts for baseline offsets in some Sentinel-2 products; reads GeoTIFF tag if blank.       | Blank / auto        |
| **Inference tile size**         | How the program divides a large image into model-sized patches; small tiles use less memory. | `128`               |
| **Tile overlap**                | Helps avoid seams between tile boundaries.                                                   | `16`                |
| **Maximum output megapixels**   | Safety limit on output size.                                                                 | `8`                 |

## Which file formats work?

- **Sentinel-2 RGB GeoTIFF (.tif/.tiff)** — recommended. The app preserves its CRS and map footprint in the enhanced TIFF. 16-bit Sentinel-2 DN inputs follow the project's normalization convention; check any reflectance offsets and band order.
- **PNG / JPG** — accepted as 8-bit RGB. Use for a simple demo, but ordinary photos are outside the training distribution, and may give poor or misleading results.
- **Clean project `.npz` pairs** — array `lr` as `(3,H,W)` and optional `hr`. If an aligned `hr` is present, PSNR and SSIM can be shown.

To download a real Sentinel-2 example, you can use the original `fetch_s2_rgb.py` supplied in your project, then upload its output TIFF. That script requires network access; inference itself runs locally.

## Important notes

- The **training scripts, datasets, and reference paper** are not needed to run this web app once you have a compatible `best.pt`.
- No PSNR or SSIM is shown for a lone LR image, because no corresponding high-resolution ground truth was given.
- The training pipeline is based on USA Sentinel-2 / NAIP pairs. Generalization to Indian imagery requires independent validation.
- TIFF pixel spacing changes in the exported metadata, but CNN output does not represent new satellite sensor measurements.
- For a genuine **3×** model, train at 3× with appropriately aligned targets, and then load that compatible checkpoint. Do not just change the output label.
- The backend runs only on `127.0.0.1` (your own computer), stores up to four results in RAM per browser session, and is not configured as a public internet service.
- Do not load checkpoints from untrusted sources. PyTorch checkpoint loading can execute unsafe serialized objects.

## Test the code

```cmd
python -m pip install pytest httpx
python -m pytest -q tests
```

Some tests create temporary, **untrained** weights to check that the processing pipeline and API function correctly. This does not verify the quality of your actual trained model.

## Troubleshooting

| Problem                                     | Fix                                                                                                       |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `py -3.11` says no Python runtime           | Run `python --version` / `py --list`; install supported Python or use the Python you already have.        |
| `.venv\\Scripts\\Activate.ps1` fails in CMD | Use `.venv\\Scripts\\activate.bat` after successfully creating `.venv`.                                   |
| `No module named fastapi`                   | Make sure your virtual environment is active, then `python -m pip install -r requirements-dashboard.txt`. |
| The webpage is empty / cannot connect       | Keep the terminal where `python app.py` is running, then visit `http://127.0.0.1:5000`.                   |
| **Model not found**                         | Put your real `best.pt` in `runs/exp1/` or correct the path under Settings.                               |
| CUDA error                                  | Switch Processing device to **CPU** / **Auto**, or install the correct CUDA build of PyTorch.             |
| Image too large                             | Crop the TIFF or adjust the output megapixel limit responsibly; very large images need more RAM.          |
| Image looks strange or dark                 | Check RGB band ordering, input type, reflectance max and offset.                                          |
| No PSNR/SSIM                                | Provide a held-out `.npz` file with both LR and aligned HR data.                                          |

## File layout

```text
DeBlur_Dashboard/
├── app.py                 # NEW: local FastAPI inference server
├── templates/index.html   # NEW: browser dashboard markup
├── static/styles.css      # NEW: responsive purple-and-pastel dashboard
├── static/app.js          # NEW: upload, navigation, slider, downloads
├── sr_backend.py          # Existing prediction integration layer
├── model.py               # Your original CNN architecture
├── evaluate.py            # Your original checkpoint loader
├── predict.py             # Your original tiled inference and TIFF export
├── train.py               # Your original training/metric helpers
├── download_data.py, clean_data.py, fetch_s2_rgb.py, run_pipeline.py
├── requirements-dashboard.txt
├── requirements.txt
├── start_windows.bat
├── start_linux.sh
├── tests/
└── runs/exp1/            # Add the best.pt from YOUR training here
```
IDSE Project files drive link :  https://drive.google.com/drive/folders/1NCiJnFW7cMSwlWhJ0eqBGq1En-_ogYGt?usp=sharing
