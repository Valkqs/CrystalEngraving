"""FastAPI server for crystal voxel engraving."""

from __future__ import annotations

import asyncio
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voxel import generate_voxels

app = FastAPI(title="Crystal Voxel Engraving")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, Dict[str, Any]] = {}
JOB_TTL_SECONDS = 60 * 60 * 2

_thread_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


def _job_prefix(job_id: str) -> str:
    return f"[JOB {job_id[:8]}]"


def _log_job_progress(job_id: str, progress: float, stage: str, detail: Optional[Dict[str, Any]] = None) -> None:
    job = jobs.get(job_id)
    if not job:
        return

    detail = detail or {}
    percent = int(max(0.0, min(1.0, float(progress))) * 100)
    last_stage = job.get("log_stage")
    last_step = job.get("log_step")
    last_log_time = job.get("log_time", 0.0)
    step = detail.get("step") if isinstance(detail, dict) else None
    max_steps = detail.get("max_steps") if isinstance(detail, dict) else None
    now = time.time()

    should_log = False
    if stage != last_stage:
        should_log = True
    elif now - last_log_time >= 2.0:
        should_log = True
    elif isinstance(step, int) and isinstance(max_steps, int):
        if last_step is None or step >= last_step + max(500, max_steps // 20):
            should_log = True

    if not should_log:
        return

    job["log_time"] = now

    extras = []
    if isinstance(step, int) and isinstance(max_steps, int):
        extras.append(f"step={step}/{max_steps}")
    if detail.get("best_f1") is not None:
        extras.append(f"best_f1={float(detail['best_f1']):.4f}")
    if detail.get("best_chaos") is not None:
        extras.append(f"best_chaos={float(detail['best_chaos']):.4f}")
    if detail.get("best_count") is not None:
        extras.append(f"best_count={int(detail['best_count'])}")
    if detail.get("accepted") is not None:
        extras.append(f"accepted={int(detail['accepted'])}")
    if detail.get("temperature") is not None:
        extras.append(f"T={float(detail['temperature']):.5f}")
    if detail.get("count_full") is not None:
        extras.append(f"count_full={int(detail['count_full'])}")
    if detail.get("count_after_sparsify") is not None:
        extras.append(f"count_after_sparsify={int(detail['count_after_sparsify'])}")
    if detail.get("point_count") is not None:
        extras.append(f"point_count={int(detail['point_count'])}")
    if detail.get("num_chains") is not None:
        extras.append(f"chains={detail['num_chains']}")

    suffix = f" | {' '.join(extras)}" if extras else ""
    print(f"{_job_prefix(job_id)} {percent}% {stage}{suffix}")
    job["log_stage"] = stage
    if isinstance(step, int):
        job["log_step"] = step


def _cleanup_jobs() -> None:
    now = time.time()
    stale_ids = [
        job_id
        for job_id, job in jobs.items()
        if now - float(job.get("updated_at", job.get("created_at", now))) > JOB_TTL_SECONDS
    ]
    for job_id in stale_ids:
        jobs.pop(job_id, None)


def _set_job_progress(job_id: str, progress: float, stage: str, detail: Optional[Dict[str, Any]] = None) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    job["progress"] = max(0.0, min(1.0, float(progress)))
    job["stage"] = stage
    job["detail"] = detail or {}
    if job.get("status") == "queued":
        job["status"] = "running"
    job["updated_at"] = time.time()
    _log_job_progress(job_id, progress, stage, detail)


def _run_generate_sync(
    front_bytes: bytes,
    side_bytes: bytes,
    size: int,
    threshold_val: Optional[int],
    invert_val: Optional[bool],
    dilate: int,
    align_x: bool,
    clean_mask: bool,
    overlap_dilate: int,
    depth_face_bridge: bool,
    close_side_z_gaps: int,
    density: float,
    uniform_strength: float,
    detail_mode: bool,
    optimize: bool,
    chaos_penalty: float,
    min_f1: float,
    sa_steps: int,
    weight_volume: float,
    rng_seed: int,
    job_id: str,
) -> Any:
    """Module-level function (picklable) — runs generate_voxels and calls back progress."""

    def progress_callback(progress: float, stage: str, detail: Optional[Dict[str, Any]] = None) -> None:
        _set_job_progress(job_id, progress, stage, detail)

    return generate_voxels(
        front_bytes,
        side_bytes,
        size=size,
        threshold=threshold_val,
        invert=invert_val,
        dilate=dilate,
        align_x=align_x,
        clean_mask=clean_mask,
        overlap_dilate=overlap_dilate,
        depth_face_bridge=depth_face_bridge,
        close_side_z_gaps=close_side_z_gaps,
        density=density,
        uniform_strength=uniform_strength,
        detail_mode=detail_mode,
        optimize=optimize,
        chaos_penalty=chaos_penalty,
        min_f1=min_f1,
        sa_steps=sa_steps,
        weight_volume=weight_volume,
        rng_seed=rng_seed,
        progress_callback=progress_callback,
    )


async def _run_generate_job(job_id: str, params: Dict[str, Any]) -> None:
    nw = os.cpu_count() or 4
    print(f"{_job_prefix(job_id)} started size={params['size']} sa_steps={params['sa_steps']} density={params['density']:.2f} chaos_penalty={params['chaos_penalty']:.2f} workers={nw}")

    try:
        _set_job_progress(job_id, 0.01, "任务已创建，正在并行计算")
        result = await asyncio.get_event_loop().run_in_executor(
            _thread_executor,
            _run_generate_sync,
            params["front_bytes"],
            params["side_bytes"],
            params["size"],
            params["threshold_val"],
            params["invert_val"],
            params["dilate"],
            params["align_x"],
            params["clean_mask"],
            params["overlap_dilate"],
            params["depth_face_bridge"],
            params["close_side_z_gaps"],
            params["density"],
            params["uniform_strength"],
            params["detail_mode"],
            params["optimize"],
            params["chaos_penalty"],
            params["min_f1"],
            params["sa_steps"],
            params["weight_volume"],
            params["rng_seed"],
            job_id,
        )
        jobs[job_id]["result"] = {
            "size": result.size,
            "count": result.count,
            "count_full": result.count_full,
            "points": result.points,
            "projection_front": result.projection_front,
            "projection_side": result.projection_side,
            "threshold_front": result.threshold_front,
            "threshold_side": result.threshold_side,
            "invert_front": result.invert_front,
            "invert_side": result.invert_side,
            "mask_front": result.mask_front,
            "mask_side": result.mask_side,
            "f1_front": result.f1_front,
            "f1_side": result.f1_side,
            "f1_total": result.f1_total,
            "chaos": result.chaos,
            "objective": result.objective,
            "optimize_params": result.optimize_params,
        }
        jobs[job_id]["status"] = "completed"
        _set_job_progress(job_id, 1.0, "生成完成，可以加载结果")
        print(f"{_job_prefix(job_id)} completed count={result.count} full={result.count_full} f1={result.f1_total:.4f} chaos={result.chaos:.4f}")
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
        jobs[job_id]["traceback"] = traceback.format_exc()
        _set_job_progress(job_id, jobs[job_id].get("progress", 0.0), "生成失败")
        print(f"{_job_prefix(job_id)} failed: {exc}")


@app.post("/api/generate")
async def api_generate(
    image_front: UploadFile = File(...),
    image_side: UploadFile = File(...),
    size: int = Form(192),
    threshold: str = Form(""),
    invert: str = Form(""),
    auto_threshold: bool = Form(True),
    dilate: int = Form(0),
    align_x: bool = Form(True),
    clean_mask: bool = Form(False),
    overlap_dilate: int = Form(1),
    depth_face_bridge: bool = Form(True),
    close_side_z_gaps: int = Form(2),
    density: float = Form(0.75),
    uniform_strength: float = Form(0.25),
    detail_mode: bool = Form(True),
    optimize: bool = Form(True),
    chaos_penalty: float = Form(0.5),
    min_f1: float = Form(0.72),
    sa_steps: int = Form(12000),
    weight_volume: float = Form(0.10),
    rng_seed: int = Form(42),
):
    _cleanup_jobs()

    front_bytes = await image_front.read()
    side_bytes = await image_side.read()

    threshold_val: Optional[int] = None
    if not auto_threshold and threshold.strip().isdigit():
        threshold_val = int(threshold)

    invert_val: Optional[bool] = None
    if invert.strip().lower() == "true":
        invert_val = True
    elif invert.strip().lower() == "false":
        invert_val = False

    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "stage": "任务排队中",
        "detail": {},
        "result": None,
        "error": None,
        "traceback": None,
        "log_stage": None,
        "log_step": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    asyncio.create_task(
        _run_generate_job(
            job_id,
            {
                "front_bytes": front_bytes,
                "side_bytes": side_bytes,
                "size": size,
                "threshold_val": threshold_val,
                "invert_val": invert_val,
                "dilate": dilate,
                "align_x": align_x,
                "clean_mask": clean_mask,
                "overlap_dilate": overlap_dilate,
                "depth_face_bridge": depth_face_bridge,
                "close_side_z_gaps": close_side_z_gaps,
                "density": density,
                "uniform_strength": uniform_strength,
                "detail_mode": detail_mode,
                "optimize": optimize,
                "chaos_penalty": chaos_penalty,
                "min_f1": min_f1,
                "sa_steps": sa_steps,
                "weight_volume": weight_volume,
                "rng_seed": rng_seed,
            },
        )
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/generate/{job_id}")
async def api_generate_status(job_id: str):
    _cleanup_jobs()
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "stage": job.get("stage", ""),
        "detail": job.get("detail", {}),
        "error": job.get("error"),
    }


@app.get("/api/generate/{job_id}/result")
async def api_generate_result(job_id: str):
    _cleanup_jobs()
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error") or "generation failed")
    if job["status"] != "completed" or job.get("result") is None:
        raise HTTPException(status_code=409, detail="job not completed")
    return job["result"]


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    dev_reload = os.getenv("CRYSTAL_SERVER_RELOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
    try:
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=dev_reload)
    finally:
        _thread_executor.shutdown(wait=True, cancel_futures=False)
