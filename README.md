# CycleGAN vs Spectral-CycleGAN + Input-Space UDA (Office-Home Art->Real)

## 1) Task Summary

This project focuses on **Office-Home: Art -> Real World** only.

- Task I:
  Compare style transfer in:
  - Spatial space (standard CycleGAN)
  - Spectral space (low-frequency band adaptation + CycleGAN)
- Task II:
  Benchmark input-space UDA methods:
  - Source-only
  - CycleGAN-augmented training
  - Spectral-CycleGAN-augmented training
  - CyCADA-inspired pipeline (simplified baseline scaffold)
  - FDA (low-frequency spectrum swapping)

## 2) Why only Office-Home code in this repo

Different datasets have different data formats and preprocessing requirements.

Examples:
- MNIST/USPS: grayscale, 1-channel, 28x28.
- SVHN/MNIST: RGB vs grayscale mismatch, resolution mismatch.
- Office-31/Office-Home/PACS: RGB photos, folder-based class structure, higher resolution.

So, this codebase is intentionally restricted to:
- Dataset pair: **Office-Home Art -> Real World**
- Unified 3-channel preprocessing and classifier setup for this pair.

## 3) Expected dataset layout

Place Office-Home under:

```text
./datasets/OfficeHome/
  Art/
    <class_name>/*.jpg
  Real World/
    <class_name>/*.jpg
```

## 4) Installation (Local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 5) Run experiments (Local CLI)

### Run all experiments

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1
```

### Run Task I only

```bash
python src/train_task1_style.py --config src/configs/officehome_art2real.yaml --mode spatial
python src/train_task1_style.py --config src/configs/officehome_art2real.yaml --mode spectral
```

### Run Task II only

```bash
python src/train_task2_uda.py --config src/configs/officehome_art2real.yaml --strategy source_only
python src/train_task2_uda.py --config src/configs/officehome_art2real.yaml --strategy cyclegan
python src/train_task2_uda.py --config src/configs/officehome_art2real.yaml --strategy spectral_cyclegan
python src/train_task2_uda.py --config src/configs/officehome_art2real.yaml --strategy cycada
python src/train_task2_uda.py --config src/configs/officehome_art2real.yaml --strategy fda
```

### Evaluate a trained classifier

```bash
python src/eval.py --config src/configs/officehome_art2real.yaml --checkpoint checkpoints/task2/fda_classifier_art2real.pt
```

## 6) Run experiments (API)

### Start API service

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1 -Port 8000
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Trigger Task I run

```bash
curl -X POST http://127.0.0.1:8000/run/task1 -H "Content-Type: application/json" -d "{\"mode\":\"spatial\"}"
curl -X POST http://127.0.0.1:8000/run/task1 -H "Content-Type: application/json" -d "{\"mode\":\"spectral\"}"
```

### Trigger Task II run

```bash
curl -X POST http://127.0.0.1:8000/run/task2 -H "Content-Type: application/json" -d "{\"strategy\":\"source_only\"}"
curl -X POST http://127.0.0.1:8000/run/task2 -H "Content-Type: application/json" -d "{\"strategy\":\"cyclegan\"}"
curl -X POST http://127.0.0.1:8000/run/task2 -H "Content-Type: application/json" -d "{\"strategy\":\"spectral_cyclegan\"}"
curl -X POST http://127.0.0.1:8000/run/task2 -H "Content-Type: application/json" -d "{\"strategy\":\"cycada\"}"
curl -X POST http://127.0.0.1:8000/run/task2 -H "Content-Type: application/json" -d "{\"strategy\":\"fda\"}"
```

### List generated checkpoints

```bash
curl http://127.0.0.1:8000/artifacts
```

## 7) Main code files

- `src/data/officehome.py`: Office-Home loader
- `src/utils/fourier.py`: FDA low-frequency replacement
- `src/models/cyclegan.py`: Minimal CycleGAN scaffold
- `src/train_task1_style.py`: Task I entry
- `src/train_task2_uda.py`: Task II entry
- `src/api/server.py`: FastAPI runner

## 8) Notes on this framework

- This is a **research scaffold** for your report experiments.
- It is runnable, but intentionally lightweight.
- You can replace lightweight generators/discriminators with full CycleGAN backbones and add full CyCADA losses for stronger performance.
