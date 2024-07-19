from typing import Optional, Union, List
from fastapi import Depends, APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, extract_frames, ImageResponse
from PIL import Image
import logging
import random
import json
import os

class HTTPError(BaseModel):
    detail: str

router = APIRouter()

logger = logging.getLogger(__name__)

responses = {
    400: {"content": {"application/json": {"schema": HTTPError.schema()}}},
    500: {"content": {"application/json": {"schema": HTTPError.schema()}}},
    # 200: {
    #     "content": {
    #         "video/mp4": {},
    #         "application/json": {"schema": VideoResponse.schema()},
    #     }
    # }
}

@router.post("/lipsync", response_model=ImageResponse, responses=responses)
async def lipsync(
    text_input: Optional[str] = Form(None),
    audio: UploadFile = File(None),
    image: UploadFile = File(...),
    return_frames: Optional[bool] = Form(False, description="Set to True to return frames instead of mp4"),
    pipeline: Pipeline = Depends(get_pipeline),
):
    if not text_input and not audio:
        raise HTTPException(status_code=400, detail="Either text_input or audio must be provided")
    
    if audio is not None:
        audio_file = audio.file
    else:
        audio_file = None

    if image is None or image.file is None:
        raise HTTPException(status_code=400, detail="Image file must be provided")


    try:
        img = pipeline(
            text_input,
            audio_file,
            image.file
        )
    except Exception as e:
        logger.error(f"LipsyncPipeline error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal Server Error: {str(e)}"
            },
        )

    seed = random.randint(0, 2**32 - 1)
    output_images = []
    output_images.append(
        {"url": image_to_data_url(img), "seed": 123, "nsfw": False }
    )

    return {"images": output_images}