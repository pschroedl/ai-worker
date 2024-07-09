from fastapi import Depends, APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
import logging
import random
import json
import os

class HTTPError(BaseModel):
    detail: str

class TextInput(BaseModel):
    text: str

router = APIRouter()

logger = logging.getLogger(__name__)

responses = {400: {"content": HTTPError}, 500: {"content": HTTPError}, 200: {"content:": {"video/mp4": {}}}}

@router.post("/lipsync", responses=responses)
async def lipsync(
    text_input: str = Form(...),
    image: UploadFile = File(...),
    model_id: str = Form(""),
    seed: int = Form(None),
    height: int = Form(576),
    width: int = Form(1024),
    fps: int = Form(6),
    motion_bucket_id: int = Form(127),
    noise_aug_strength: float = Form(0.02),
    pipeline: Pipeline = Depends(get_pipeline),
):

    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"pipeline configured with {pipeline.model_id} but called with {model_id}"
            },
        )

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

   
    try:
        output_video_path = pipeline(
            text_input,
            image.file,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"LipsyncPipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal Server Error: {str(e)}"
            },
        )

    if os.path.exists(output_video_path):
        return FileResponse(path=output_video_path, media_type='video/mp4', filename="lipsync_video.mp4")
    else:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"no output found for {output_video_path}"
            },
        )
