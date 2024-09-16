import uuid
from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir, SafetyChecker, save_image_to_temp_file
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import subprocess
import os
import torch
import logging
import time
import gc
from tqdm import tqdm



logger = logging.getLogger(__name__)

class LipsyncPipeline(Pipeline):
    def __init__(self):
        self.device = get_torch_device()
        # Load FastSpeech 2 and HiFi-GAN models
        self.TTS_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(self.device)

        self.TTS_tokenizer = AutoTokenizer.from_pretrained(
            "parler-tts/parler-tts-large-v1",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # self.TTS_tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer", cache_dir=get_model_dir())
        # self.TTS_model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer", cache_dir=get_model_dir()).to(self.device)
        # self.TTS_hifigan = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan", cache_dir=get_model_dir()).to(self.device)


    def __call__(self, text, audio_file, image_file):
        # Save Source Image to Disk
        temp_image_file_path = save_image_to_temp_file(image_file)

        # generate unique filename
        unique_audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join("/tmp/", unique_audio_filename)

        if audio_file is None:
            self.generate_speech(text, audio_path)
        else: 
            with open(audio_path, 'wb') as f:
                f.write(audio_file.read())

        # Generate LipSync
        lipsync_output_path = self.generate_real3d_lipsync(temp_image_file_path, audio_path, "/app/output")

        return lipsync_output_path


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

        # # Construct the command to run the shell script
        # command = ["/bin/bash",shell_script_path, image_path, audio_path, output_video_path, pose_drv]

        # real3dportrait_path = "/models/models--yerfor--Real3DPortrait/"
        # os.chdir(real3dportrait_path)
        # print(f"Running command: {' '.join(command)}")
        # subprocess.run(command, check=True)

        # Path to the Real3DPortrait Python script
        real3dportrait_script = "/models/models--yerfor--Real3DPortrait/inference/real3d_infer.py"
        
        # Virtual environment Python binary
        python_bin = "/opt/lipsync_venv/bin/python"
        
        # Define the PYTHONPATH
        pythonpath = "/models/models--yerfor--Real3DPortrait/"

        # Environment variables (you can add others if needed)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{pythonpath}:{env.get('PYTHONPATH', '')}"

        # Construct the command to run the inference script
        command = [
            python_bin,
            real3dportrait_script,
            "--src_img", image_path,
            "--drv_aud", audio_path,
            "--out_name", output_video_path,
            "--drv_pose", pose_drv,
            "--out_mode", "final"
        ]

        # Change to the appropriate directory
        real3dportrait_path = "/models/models--yerfor--Real3DPortrait/"
        os.chdir(real3dportrait_path)

        print(f"Running command: {' '.join(command)}")

        # Run the command using subprocess, passing the environment and the command
        subprocess.run(command, env=env, check=True)

        # Check if the output video was created
        if not os.path.exists(output_video_path):
            raise FileNotFoundError(f"Cannot find the output video file: {output_video_path}")
        
        print("Lip-sync video generation complete.")
        return output_video_path

    def generate_speech(self, text, output_file_name):
        # Tokenize input text
        # inputs = self.TTS_tokenizer(text, return_tensors="pt").to(self.device)
        
        # # Ensure input IDs remain in Long tensor type
        # input_ids = inputs["input_ids"].to(self.device)
        
        # # Generate spectrogram
        # output_dict = self.TTS_model(input_ids, return_dict=True)
        # spectrogram = output_dict["spectrogram"]

        # # Convert spectrogram to waveform
        # waveform = self.TTS_hifigan(spectrogram)

        description = "A male speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

        input_ids = self.TTS_tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.TTS_tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        generation = self.TTS_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        generated_audio = generation.cpu().numpy().squeeze()

        sf.write(output_file_name, generated_audio, samplerate=44100)
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
