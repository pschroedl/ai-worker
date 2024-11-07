import os
import logging
import torch
import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel
import hashlib
from omegaconf import OmegaConf

base_sam2_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "segment-anything-2-real-time"))
sys.path.append(base_sam2_dir)
from sam2.build_sam import build_sam2_camera_predictor

from .interface import Pipeline

def make_sam2_path(rel_path):
    return os.path.normpath(os.path.join(base_sam2_dir, rel_path))


class Sam2InferParams(BaseModel):
    class Config:
        extra = 'forbid'

    point_prompt_x: float = 0.67  # Relative X position for point prompt (e.g., 2/3 from left)
    point_prompt_y: float = 0.5   # Relative Y position for point prompt (e.g., center vertically)
    apply_postprocessing: bool = True

    def to_omegaconf(self) -> OmegaConf:
        return OmegaConf.create(self.dict())

class Sam2Params(BaseModel):
    class Config:
        extra = 'forbid'

    checkpoint: str = 'sam2_hiera_small.pt'
    infer_params: Sam2InferParams = Sam2InferParams()

base_pipe_config_path = make_sam2_path('configs/sam2_hiera_s.yaml')

class Sam2Pipeline(Pipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self.pipe = None
        self.update_params(**params)

    def process_frame(self, image: Image.Image) -> Image.Image:
        # Convert PIL image to OpenCV BGR format
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Prepare point prompts for SAM2
        height, width = cv2_image.shape[:2]
        point_x = int(self.params.infer_params.point_prompt_x * width)
        point_y = int(self.params.infer_params.point_prompt_y * height)
        point = [point_x, point_y]
        points = [point]
        labels = [1]  # Positive prompt

        # Track the object in the frame using SAM2
        if self.first_frame:
            self.pipe.load_first_frame(cv2_image)
            obj_id = 1
            frame_idx = 0
            _, _, out_mask_logits = self.pipe.add_new_prompt(frame_idx, obj_id, points=points, labels=labels)
            self.first_frame = False
        else:
            _, out_mask_logits = self.pipe.track(cv2_image)

        # Process output mask only if it's non-empty
        if out_mask_logits.shape[0] > 0:
            mask = (out_mask_logits[0, 0] > 0).cpu().numpy().astype("uint8") * 255
        else:
            mask = np.zeros((height, width), dtype="uint8")

        # Prepare the mask for overlay
        inverted_mask_colored = cv2.cvtColor(cv2.bitwise_not(mask), cv2.COLOR_GRAY2BGR)
        overlayed_frame = cv2.addWeighted(cv2_image, 0.7, inverted_mask_colored, 0.3, 0)

        # Convert the processed frame back to PIL format
        return Image.fromarray(cv2.cvtColor(overlayed_frame, cv2.COLOR_BGR2RGB))

    def update_params(self, **params):
        new_params = Sam2Params(**params)
        if not os.path.isabs(new_params.checkpoint):
            new_params.checkpoint = make_sam2_path(f"checkpoints/{new_params.checkpoint}")

        # Load configuration file and merge parameters
        new_cfg = OmegaConf.load(base_pipe_config_path)
        new_cfg.infer_params = OmegaConf.merge(new_cfg.infer_params, new_params.infer_params.to_omegaconf())

        new_pipe = build_sam2_camera_predictor(
            config_file=base_pipe_config_path,
            ckpt_path=new_params.checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
            mode="eval",
            apply_postprocessing=new_params.infer_params.apply_postprocessing,
        )

        if self.pipe is not None:
            self.pipe.clean_models()

        self.params = new_params
        self.cfg = new_cfg
        self.pipe = new_pipe
        self.first_frame = True

