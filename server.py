"""FastAPI server for crystal voxel engraving."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from voxel import generate_voxels
from image_preprocess import extract_mask, load_grayscale_fit
import numpy as np

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
    optimizer_algo: str,
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
        optimizer_algo=optimizer_algo,
    )


async def _run_generate_job(job_id: str, params: Dict[str, Any]) -> None:
    nw = os.cpu_count() or 4
    print(f"{_job_prefix(job_id)} started size={params['size']} algo={params['optimizer_algo']} sa_steps={params['sa_steps']} density={params['density']:.2f} chaos_penalty={params['chaos_penalty']:.2f} workers={nw}")

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
            params["optimizer_algo"],
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
    optimizer_algo: str = Form("fast"),
):
    _cleanup_jobs()

    front_bytes = await image_front.read()
    side_bytes = await image_side.read()
    print(f"[DEBUG] /api/generate received: front_size={len(front_bytes)} side_size={len(side_bytes)} size={size} density={density} chaos_penalty={chaos_penalty} min_f1={min_f1} sa_steps={sa_steps} optimize={optimize} optimizer_algo={optimizer_algo}")

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
                "optimizer_algo": optimizer_algo,
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
        print(f"[DEBUG] /api/generate/{job_id}/result → 404 job not found")
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] == "failed":
        print(f"[DEBUG] /api/generate/{job_id}/result → 500 failed: {job.get('error')}")
        raise HTTPException(status_code=500, detail=job.get("error") or "generation failed")
    if job["status"] != "completed" or job.get("result") is None:
        print(f"[DEBUG] /api/generate/{job_id}/result → 409 not completed, status={job['status']}")
        raise HTTPException(status_code=409, detail="job not completed")
    result = job["result"]
    print(f"[DEBUG] /api/generate/{job_id}/result → 200 count={result.get('count')} points_len={len(result.get('points', []))} size={result.get('size')}")
    return result


@app.get("/")
async def index():
    return FileResponse("static/index.html")


def _render_mask_to_base64(gray: np.ndarray, threshold: int, invert: bool) -> str:
    """Render a binary mask as a base64 PNG (black/green tint for front, black/blue for side)."""
    mask = gray <= threshold if invert else gray >= threshold
    arr = mask.astype(np.uint8) * 255
    # Tint: foreground gets a color so it's visible against black background
    rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
    rgba[..., 3] = arr  # alpha
    rgba[arr > 0] = [80, 200, 120, 255]  # green tint for foreground
    rgba[arr == 0, 3] = 0  # background transparent
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _compute_preview(
    front_bytes: bytes,
    side_bytes: bytes,
    size: int,
    threshold_str: str,
    invert_str: str,
    auto_threshold: bool,
) -> Dict[str, Any]:
    """Compute mask previews and return base64 PNGs + diagnostics."""
    print(f"[PREVIEW] front_size={len(front_bytes)} side_size={len(side_bytes)} "
          f"size={size} threshold={threshold_str} invert={invert_str} auto={auto_threshold}")

    preview_size = 160
    front_gray = load_grayscale_fit(front_bytes, preview_size)
    side_gray = load_grayscale_fit(side_bytes, preview_size)
    print(f"[PREVIEW] front_gray shape={front_gray.shape} range=[{front_gray.min()}, {front_gray.max()}]")
    side_gray = load_grayscale_fit(side_bytes, preview_size)
    print(f"[PREVIEW] side_gray shape={side_gray.shape} range=[{side_gray.min()}, {side_gray.max()}]")

    print("[PREVIEW] calling extract_mask for front...")
    threshold_val: Optional[int] = None
    if not auto_threshold and threshold_str.strip().isdigit():
        threshold_val = int(threshold_str)
    invert_val: Optional[bool] = None
    if invert_str.strip().lower() == "true":
        invert_val = True
    elif invert_str.strip().lower() == "false":
        invert_val = False
    f_mask, t_front, inv_front = extract_mask(front_gray, threshold_val, invert_val)
    print(f"[PREVIEW] front extract_mask done")
    s_mask, t_side, inv_side = extract_mask(side_gray, threshold_val, invert_val)
    print(f"[PREVIEW] side extract_mask done")
    print(f"[PREVIEW] front: threshold={t_front} invert={inv_front} mask_true={f_mask.sum()} / {f_mask.size}")
    print(f"[PREVIEW] side:  threshold={t_side} invert={inv_side} mask_true={s_mask.sum()} / {s_mask.size}")

    print("[PREVIEW] calling _render_mask_to_base64 for front...")
    front_png = _render_mask_to_base64(front_gray, t_front, inv_front)
    side_png = _render_mask_to_base64(side_gray, t_side, inv_side)
    print("[PREVIEW] _render_mask_to_base64 done")

    return {
        "front_png": front_png,
        "side_png": side_png,
        "threshold_front": t_front,
        "threshold_side": t_side,
        "invert_front": inv_front,
        "invert_side": inv_side,
        "mask_front_ratio": float(f_mask.sum()) / f_mask.size if f_mask.size > 0 else 0,
        "mask_side_ratio": float(s_mask.sum()) / s_mask.size if s_mask.size > 0 else 0,
    }


@app.post("/api/preview")
async def api_preview(
    image_front: UploadFile = File(...),
    image_side: UploadFile = File(...),
    size: int = Form(192),
    threshold: str = Form(""),
    invert: str = Form(""),
    auto_threshold: bool = Form(True),
):
    front_bytes = await image_front.read()
    side_bytes = await image_side.read()
    print(f"[PREVIEW] POST received: front={image_front.filename} side={image_side.filename}")

    try:
        result = _compute_preview(front_bytes, side_bytes, size, threshold, invert, auto_threshold)
        print(f"[PREVIEW] done, front_png len={len(result['front_png'])}")
        return result
    except Exception as exc:
        print(f"[PREVIEW] ERROR: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/export/{job_id}/ply")
async def api_export_ply(job_id: str):
    """Export voxel point cloud as PLY file for 3D printing / CAD import."""
    _cleanup_jobs()
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    result = job.get("result")
    if not result:
        raise HTTPException(status_code=404, detail="result not ready")

    points = result.get("points", [])
    size = result.get("size", 192)

    lines = ["ply"]
    lines.append("format ascii 1.0")
    lines.append(f"element vertex {len(points)}")
    lines.append("property float x")
    lines.append("property float y")
    lines.append("property float z")
    lines.append("end_header")

    half = size / 2.0
    for pt in points:
        x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
        lines.append(f"{(x - half):.4f} {(y - half):.4f} {(z - half):.4f}")

    content = "\n".join(lines)
    from starlette.responses import Response
    return Response(
        content=content,
        media_type="model/ply",
        headers={"Content-Disposition": f"attachment; filename=crystal_{job_id[:8]}.ply"},
    )


if __name__ == "__main__":
    dev_reload = os.getenv("CRYSTAL_SERVER_RELOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
    try:
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=dev_reload)
    finally:
        _thread_executor.shutdown(wait=True, cancel_futures=False)
