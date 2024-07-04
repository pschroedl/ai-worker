from fastapi import Depends, APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
import logging
import random

class HTTPError(BaseModel):
    detail: str

class VideoResponse(BaseModel):
    video_url: str

router = APIRouter()

logger = logging.getLogger(__name__)

responses = {400: {"model": HTTPError}, 500: {"model": HTTPError}}

@router.post("/lipsync", response_model=VideoResponse, responses=responses)
async def lipsync(
    text: UploadFile = File(...),
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
            text.file,
            image.file,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"LipsyncPipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500, content={"detail": "LipsyncPipeline error"}
        )

    return {"video_url": output_video_path}
