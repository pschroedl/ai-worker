import uuid
from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir, SafetyChecker
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
import soundfile as sf
from huggingface_hub import login
import subprocess
import os
import torch
import logging
import time
import gc
from tqdm import tqdm
from PIL import Image, ImageFile
import tempfile


logger = logging.getLogger(__name__)

class LipsyncPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id # not conforming to this pattern at the moment
        self.device = get_torch_device()
        self.generation_id = 0;
        # Authenticate with Hugging Face    
        self.authenticate_huggingface()

    def authenticate_huggingface(self):
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            login(token=token)
        else:
            raise ValueError("Hugging Face token not found. Set HUGGINGFACE_TOKEN environment variable.")

    def __call__(self, text, image_file, seed=None, model_id="real3dportrait"):
        self.model_id = model_id

        # Save Source Image to Disk
        temp_image_file_path = save_image_to_temp_file(image_file)

        # Generate Voice
        audio_path = self.generate_speech(text)

        # Generate LipSync
        lipsync_output_path = self.generate_real3d_lipsync(temp_image_file_path, audio_path, "/app/output")

        return lipsync_output_path

    def generate_real3d_lipsync(self, image_path, audio_path, output_path):
        
        real3dportrait_path = "/models/models--yerfor--Real3DPortrait/"
        unique_video_filename = generate_unique_filename(output_path, "mp4")
        output_video_path = os.path.join(output_path, unique_video_filename)

        os.chdir(real3dportrait_path)
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Path to the shell script
        shell_script_path = "/models/models--yerfor--Real3DPortrait/run_real3dportrait.sh"

        # Construct the command to run the shell script
        command = [shell_script_path, image_path, audio_path, os.path.join(output_path, output_video_path)]

        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)

        # Check if the output video was created
        if not os.path.exists(output_video_path):
            raise FileNotFoundError(f"Cannot find the output video file: {output_video_path}")
        
        print("Lip-sync video generation complete.")
        return output_video_path

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

        unique_audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join("/tmp/", unique_audio_filename)
        sf.write(audio_path, waveform.squeeze().detach().cpu().numpy(), samplerate=22050)
        return audio_path

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

    def merge_audio_video(self, video_path, audio_path, output_path):
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
        # Construct the ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copy the video codec
            '-c:a', 'aac',   # Use AAC for audio
            '-strict', 'experimental',
            output_path
        ]
    
        # Run the ffmpeg command
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully merged audio and video to: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error merging audio and video: {e}")
            print(f"ffmpeg stdout: {e.stdout}")
            print(f"ffmpeg stderr: {e.stderr}")

def save_image_to_temp_file(image_file):
    image = Image.open(image_file)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file, format="JPEG")
    temp_file.close()
    return temp_file.name