from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir, SafetyChecker
from phonemizer import phonemize
from phonemizer.separator import Separator
from diffusers import StableVideoDiffusionPipeline
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, FastSpeech2ForConditionalGeneration, FastSpeech2Tokenizer
from scipy.io.wavfile import write
from huggingface_hub import HfFolder, login, hf_hub_download
import subprocess
import torch
import os
import logging
import time
from PIL import Image, ImageFile
from fastapi import Form
from fastapi.responses import JSONResponse
from typing import List, Tuple, Optional

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class LipsyncPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = get_torch_device()
        
        # Authenticate with Hugging Face
        self.authenticate_huggingface()
        
        # Load Wav2Lip model and inference script
        self.wav2lip_model_path = hf_hub_download(repo_id="camenduru/Wav2Lip", filename="checkpoints/wav2lip.pth", cache_dir=get_model_dir())
        #this for some reason is not working as expected
        self.wav2lip_inference_script = hf_hub_download(repo_id="camenduru/Wav2Lip", filename="inference.py", cache_dir=get_model_dir())
       
       # Load FastSpeech 2 and HiFi-GAN models
        self.fastspeech2_model_path = os.path.join(get_model_dir(), "fastspeech2")
        self.fastspeech2_model = FastSpeech2ForConditionalGeneration.from_pretrained(self.fastspeech2_model_path).to(self.device)
        self.fastspeech2_tokenizer = FastSpeech2Tokenizer.from_pretrained(self.fastspeech2_model_path)
        self.hifigan_model_path = os.path.join(get_model_dir(), "hifigan")
        self.hifigan = torch.hub.load("facebookresearch/hifigan", "hifigan_v1").to(self.device)

        # Load Stable Diffusion model for image-to-video generation
        self.stable_video_model_id = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        self.stable_video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.stable_video_model_id, cache_dir=get_model_dir(), variant="fp16", torch_dtype=torch.float16
        ).to(self.device)

    def authenticate_huggingface(self):
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            login(token=token)
        else:
            raise ValueError("Hugging Face token not found. Set HUGGINGFACE_TOKEN environment variable.")

    def __call__(self, text_file, image_file, seed=None):
        # Step 1: Read the text file for transcription
        transcription = self.read_text_file(text_file)
        
        # Step 2: Generate audio from transcription using Whisper
        audio_path = self.generate_speech(transcription)
        
        # Step 3: Convert transcription to phonemes using Phonemizer
        # phonemes = self.convert_to_phonemes(transcription)
        
        # Step 4: Generate video frames using Stable Diffusion
        frames = self.generate_video_frames(image_file, seed)
        
        # Step 5: Save frames to video
        temp_video_path = self.save_frames_to_video(frames)
        
        # Step 6: Generate lip-synced video using Wav2Lip
        output_video_path = self.generate_lip_sync_video(temp_video_path, audio_path)
        
        return output_video_path

    # def read_text_file(self, text_file):
    #     with open(text_file, 'r') as file:
    #         transcription = file.read().strip()
    #     return transcription

    def read_text_file(self, text_file):
        text_file.seek(0)
        transcription = text_file.read().decode('utf-8').strip()
        return transcription

    def generate_speech(self, text):
        inputs = self.fastspeech2_tokenizer(text, return_tensors="pt").to(self.device)
        mel_outputs = self.fastspeech2_model.generate(**inputs)
        audio = self.hifigan(mel_outputs[0]).squeeze(0).cpu().numpy()
        audio_path = "output_speech.wav"
        write(audio_path, 22050, audio)
        return audio_path

    def convert_to_phonemes(self, text):
        phonemes = phonemize(
            text=text,
            language='en-us',
            backend='espeak',
            separator=Separator(phone=' ', word='|'),
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
            njobs=4
        )
        return phonemes

    def generate_video_frames(self, image_file, seed):
        image = Image.open(image_file)
        image = [image.resize((1024, 576))]
        image *= 2
        video_frames, _ = self.stable_video_pipeline(image=image, seed=seed)
        return video_frames

    def save_frames_to_video(
        self,
        batch_frames,
        height: int = 576,
        width: int = 1024,
        fps: int = 6,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        safety_check: bool = True,
        seed: Optional[int] = None
    ):
        temp_video_path = "temp_video.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-i", "-",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-s", f"{width}x{height}",
            temp_video_path
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        for frame in batch_frames:
            frame.save(process.stdin, format="JPEG")
        process.stdin.close()
        process.wait()
        return temp_video_path

    def generate_lip_sync_video(self, video_path, audio_path):
        output_video_path = "output_lipsync.mp4"
        # inference_script_path = os.path.join(get_model_dir(), "wav2lip", "inference.py")
        inference_script_path = os.path.join(get_model_dir(), "Wav2Lip", "inference.py")
        command = [
            "python", inference_script_path,
            "--checkpoint_path", self.wav2lip_model_path,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_video_path
        ]
        subprocess.run(command, check=True)
        return output_video_path

def image_to_data_url(image):
    from io import BytesIO
    import base64

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"
