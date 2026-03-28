from pathlib import Path
from subprocess import Popen
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Office-Home UDA Runner", version="0.1.0")


class Task1Req(BaseModel):
    config: str = "src/configs/officehome_art2real.yaml"
    mode: Literal["spatial", "spectral"]


class Task2Req(BaseModel):
    config: str = "src/configs/officehome_art2real.yaml"
    strategy: Literal["source_only", "cyclegan", "spectral_cyclegan", "cycada", "fda"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run/task1")
def run_task1(req: Task1Req):
    cmd = [
        "python",
        "src/train_task1_style.py",
        "--config",
        req.config,
        "--mode",
        req.mode,
    ]
    Popen(cmd)
    return {"message": "Task1 started", "cmd": cmd}


@app.post("/run/task2")
def run_task2(req: Task2Req):
    cmd = [
        "python",
        "src/train_task2_uda.py",
        "--config",
        req.config,
        "--strategy",
        req.strategy,
    ]
    Popen(cmd)
    return {"message": "Task2 started", "cmd": cmd}


@app.get("/artifacts")
def artifacts():
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return {"checkpoints": []}

    items = [str(p) for p in ckpt_dir.rglob("*.pt")]
    return {"checkpoints": items}
