import uuid
from app.pipelines.base import Pipeline
from app.routes.util import image_to_data_url, ImageResponse, HTTPError, http_error
from app.pipelines.util import get_torch_device, get_model_dir, SafetyChecker, save_image_to_temp_file
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
import soundfile as sf
import subprocess
import os
import torch
import logging
import time
import gc
from tqdm import tqdm
from PIL import Image

logger = logging.getLogger(__name__)


class LipsyncPipeline(Pipeline):
    def __init__(self):
        self.device = get_torch_device()
        # Load FastSpeech 2 and HiFi-GAN models
        # self.TTS_tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer", cache_dir=get_model_dir())
        # self.TTS_model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer", cache_dir=get_model_dir()).to(self.device)
        # self.TTS_hifigan = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan", cache_dir=get_model_dir()).to(self.device)


    def __call__(self, text, audio_file, image_file):
        # Save Source Image to Disk
        temp_image_file_path = save_image_to_temp_file(image_file)
        image = Image.open(temp_image_file_path)
        # # generate unique filename
        # unique_audio_filename = f"{uuid.uuid4()}.wav"
        # audio_path = os.path.join("/tmp/", unique_audio_filename)

        # if audio_file is None:
        #     self.generate_speech(text, audio_path)
        # else: 
        #     with open(audio_path, 'wb') as f:
        #         f.write(audio_file.read())

        # # Generate LipSync
        # lipsync_output_path = self.generate_real3d_lipsync(temp_image_file_path, audio_path, "/app/output")

        return image

    def upscale(self, input_image_path, output_path, resolution=1024, num_inference_steps=20, strength=0.2, hdr=0, guidance_scale=6, controlnet_strength=0.75, scheduler_name="DDIM"):
        # Path to the shell script
        shell_script_path = "/app/run_upscale.sh"

        # Generate unique filename
        unique_output_filename = f"{uuid.uuid4()}.png"
        output_image_path = os.path.join(output_path, unique_output_filename)

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Construct the command to run the shell script
        command = [
            shell_script_path,
            input_image_path,
            output_image_path,
            str(resolution),
            str(num_inference_steps),
            str(strength),
            str(hdr),
            str(guidance_scale),
            str(controlnet_strength),
            scheduler_name
        ]

        # Change to the directory where the script is located if needed
        script_directory = "/models/TileUpscalerV2/"
        os.chdir(script_directory)
        print(f"Running command: {' '.join(command)}")
        
        # Run the shell script
        subprocess.run(command, check=True)

        # Check if the output image was created
        if not os.path.exists(output_image_path):
            raise FileNotFoundError(f"Cannot find the output image file: {output_image_path}")
        
        print("Image processing complete.")
        return output_image_path

    def generate_real3d_lipsync(self, image_path, audio_path, output_path):

        # Path to the shell script
        shell_script_path = "/app/run_real3dportrait.sh"

        # generate unique filename
        unique_video_filename = f"{uuid.uuid4()}.mp4"
        output_video_path = os.path.join(output_path, unique_video_filename)

        # parameter for driving head pose - default in repo is a bit wonky
        pose_drv = 'static'

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Construct the command to run the shell script
        command = [shell_script_path, image_path, audio_path, output_video_path, pose_drv]

        real3dportrait_path = "/models/TileUpscalerV2/"
        os.chdir(real3dportrait_path)
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)

        # Check if the output video was created
        if not os.path.exists(output_video_path):
            raise FileNotFoundError(f"Cannot find the output video file: {output_video_path}")
        
        print("Lip-sync video generation complete.")
        return output_video_path

    def generate_speech(self, text, output_file_name):
        # Tokenize input text
        inputs = self.TTS_tokenizer(text, return_tensors="pt").to(self.device)
        
        # Ensure input IDs remain in Long tensor type
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate spectrogram
        output_dict = self.TTS_model(input_ids, return_dict=True)
        spectrogram = output_dict["spectrogram"]

        # Convert spectrogram to waveform
        waveform = self.TTS_hifigan(spectrogram)

        sf.write(output_file_name, waveform.squeeze().detach().cpu().numpy(), samplerate=22050)
        return output_file_name

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
