from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir, SafetyChecker
# from phonemizer import phonemize
# from phonemizer.separator import Separator
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
import soundfile as sf
from huggingface_hub import HfFolder, login, hf_hub_download, file_download
import subprocess
import torch
import os
import logging
import time
import gc
from PIL import Image, ImageFile
from fastapi import Form
from fastapi.responses import JSONResponse
from typing import List, Tuple, Optional

ImageFile.LOAD_TRUNCATED_IMAGES = True
SFAST_WARMUP_ITERATIONS = 2  # Model warm-up iterations when SFAST is enabled.

logger = logging.getLogger(__name__)

class LipsyncPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = get_torch_device()
        
        # Authenticate with Hugging Face
        self.authenticate_huggingface()

    def authenticate_huggingface(self):
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            login(token=token)
        else:
            raise ValueError("Hugging Face token not found. Set HUGGINGFACE_TOKEN environment variable.")

    def __call__(self, text_file, image_file, seed=None):
        # Step 1: Read the text file for transcription
        transcription = self.read_text_file(text_file)
        
        # Step 2: Generate audio from transcription using TTS
        audio_path = self.generate_speech(transcription)
        
        # Step 3: Convert transcription to phonemes using Phonemizer
        # phonemes = self.convert_to_phonemes(transcription)
        
        # Step 4: Generate video frames using Stable Diffusion
        temp_video_path = self.generate_video(image_file)
        
        # Step 6: Generate lip-synced video using Wav2Lip
        output_video_path = self.generate_lip_sync_video(temp_video_path, audio_path)
        
        return output_video_path

    def read_text_file(self, text_file):
        text_file.seek(0)
        transcription = text_file.read().decode('utf-8').strip()
        return transcription

    def generate_speech(self, text):
        # Load FastSpeech 2 and HiFi-GAN models
        self.TTS_tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer", cache_dir=get_model_dir())
        self.TTS_model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer", cache_dir=get_model_dir()).to(self.device)
        self.TTS_hifigan = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan", cache_dir=get_model_dir()).to(self.device)

        # Tokenize input text
        inputs = self.TTS_tokenizer(text, return_tensors="pt").to(self.device)
        
        # Ensure input IDs remain in Long tensor type
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate spectrogram
        output_dict = self.TTS_model(input_ids, return_dict=True)
        spectrogram = output_dict["spectrogram"]
        print("Unloading TTS Model")
        self.unload_model(self.TTS_model)

        # Convert spectrogram to waveform
        waveform = self.TTS_hifigan(spectrogram)
        print("Unloading TTS HifiGAN")
        self.unload_model(self.TTS_hifigan)

        # Save the audio
        audio_path = "output_speech.wav"
        sf.write(audio_path, waveform.squeeze().detach().cpu().numpy(), samplerate=22050)
        return audio_path


    # def convert_to_phonemes(self, text):
    #     phonemes = phonemize(
    #         text=text,
    #         language='en-us',
    #         backend='espeak',
    #         separator=Separator(phone=' ', word='|'),
    #         strip=True,
    #         preserve_punctuation=True,
    #         with_stress=True,
    #         njobs=4
    #     )
    #     return phonemes

    def generate_video(self, image_file):
        # Load Stable Diffusion model for image-to-video generation
        # self.stable_video_model_id = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        # self.stable_video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        #     self.stable_video_model_id, cache_dir=get_model_dir(), variant="fp16", torch_dtype=torch.float16
        # ).to(self.device)
    
        self.stable_video_pipeline = self.load_stable_diffusion_model()
        # self.stable_video_pipeline.en_and_decode_n_samples_a_time = (
        #             8  # Decode n frames at a time
        #         )
        global lowvram_mode
        lowvram_mode = True

        image = Image.open(image_file)
        image = [image.resize((1024, 576))]
        image *= 2
        video_frames = self.stable_video_pipeline(image=image,num_frames=25, decode_chunk_size=8).frames[0]
        
        # Unload the Stable Diffusion model from the GPU
        print("Unloading Stable Diffusion model")
        self.unload_model(self.stable_video_pipeline)
        video_path = "generated_video.mp4"
        export_to_video(video_frames, video_path, fps=7)
        return video_path

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
        # Load Wav2Lip model and inference script
        print("loading Lipsync Model")
        wav2lip_model_path = hf_hub_download(repo_id="camenduru/Wav2Lip", filename="checkpoints/wav2lip.pth", cache_dir=get_model_dir())
        wav2lip_inference_script = hf_hub_download(repo_id="camenduru/Wav2Lip", filename="inference.py", cache_dir=get_model_dir())
        
        output_video_path = "output_lipsync.mp4"
        inference_script_path = os.path.join(get_model_dir(), "Wav2Lip", "inference.py")
        command = [
            "python", inference_script_path,
            "--checkpoint_path", wav2lip_model_path,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_video_path
        ]
        subprocess.run(command, check=True)
        # TODO (PSCHROEDL) : ASYNC unload lipsync model(?)
        return output_video_path
    
    def unload_model(self, model):
        # Move all components of the pipeline to CPU if they have the .cpu() method
        if hasattr(model, 'components'):
            for component in model.components.values():
                if hasattr(component, 'cpu'):
                    component.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(20)

    
    def load_stable_diffusion_model(self):
        model_id = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = self.device
        folder_name = file_download.repo_folder_name(repo_id=model_id, repo_type="model")
        folder_path = os.path.join(get_model_dir(), folder_name)
        has_fp16_variant = any(".fp16.safetensors" in fname for _, _, files in os.walk(folder_path) for fname in files)
        
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("ImageToVideoPipeline loading fp16 variant for %s", model_id)
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        ldm = StableVideoDiffusionPipeline.from_pretrained(model_id, **kwargs)
        ldm.to(torch_device)

        sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
        deepcache_enabled = os.getenv("DEEPCACHE", "").strip().lower() == "true"
        if sfast_enabled and deepcache_enabled:
            logger.warning(
                "Both 'SFAST' and 'DEEPCACHE' are enabled. This is not recommended "
                "as it may lead to suboptimal performance. Please disable one of them."
            )

        if sfast_enabled:
            logger.info(
                "ImageToVideoPipeline will be dynamically compiled with stable-fast "
                "for %s", model_id
            )
            from app.pipelines.optim.sfast import compile_model

            ldm = compile_model(ldm)

            if os.getenv("SFAST_WARMUP", "true").lower() == "true":
                warmup_kwargs = {
                    "image": PIL.Image.new("RGB", (576, 1024)),
                    "height": 576,
                    "width": 1024,
                    "fps": 6,
                    "motion_bucket_id": 127,
                    "noise_aug_strength": 0.02,
                    "decode_chunk_size": 4,
                }

                logger.info("Warming up ImageToVideoPipeline pipeline...")
                total_time = 0
                for ii in range(SFAST_WARMUP_ITERATIONS):
                    t = time.time()
                    try:
                        ldm(**warmup_kwargs).frames
                    except Exception as e:
                        logger.error(f"ImageToVideoPipeline warmup error: {e}")
                        raise e
                    iteration_time = time.time() - t
                    total_time += iteration_time
                    logger.info(
                        "Warmup iteration %s took %s seconds", ii + 1, iteration_time
                    )
                logger.info("Total warmup time: %s seconds", total_time)

        if deepcache_enabled:
            logger.info(
                "ImageToVideoPipeline will be optimized with DeepCache for %s",
                model_id
            )
            from app.pipelines.optim.deepcache import enable_deepcache

            ldm = enable_deepcache(ldm)

        return ldm

def image_to_data_url(image):
    from io import BytesIO
    import base64

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


