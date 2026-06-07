"""FastAPI server for crystal voxel engraving."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
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
    edge_strip_fill: bool = Form(True),
    edge_wall_wrap: bool = Form(True),
    close_side_z_gaps: int = Form(2),
    density: float = Form(0.75),
    uniform_strength: float = Form(0.25),
    detail_mode: bool = Form(True),
):
    front_bytes = await image_front.read()
    side_bytes = await image_side.read()

    threshold_val: int | None = None
    if not auto_threshold and threshold.strip().isdigit():
        threshold_val = int(threshold)

    invert_val: bool | None = None
    if invert.strip().lower() == "true":
        invert_val = True
    elif invert.strip().lower() == "false":
        invert_val = False

    result = generate_voxels(
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
        edge_strip_fill=edge_strip_fill,
        edge_wall_wrap=edge_wall_wrap,
        close_side_z_gaps=close_side_z_gaps,
        density=density,
        uniform_strength=uniform_strength,
        detail_mode=detail_mode,
    )

    return {
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
    }


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
