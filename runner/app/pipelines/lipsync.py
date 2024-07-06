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
import cv2
from PIL import Image, ImageFile
import tempfile


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

    def __call__(self, text, image_file, seed=None, model_id="real3dportrait"):
        self.model_id = model_id
        # Generate Voice
        audio_path = self.generate_speech(text)
        # Generate LipSync
        temp_image_file_path = save_image_to_temp_file(image_file)
        if model_id == "real3dportrait":
            output_video_path = self.generate_real3d_lipsync(temp_image_file_path, audio_path)
        elif model_id == "wav2lip":
            output_video_path = self.generate_wav2lip_lipsync(temp_image_file_path, audio_path)
        else:
            logger.error(f"Invalid model_id: {model_id}, options: ['real3dportrait', 'wav2lip']")
            return ""
        # Enhance with ESRGAN
        HD_video_path = self.generate_HD_upscale(output_video_path)
        final_output_path = "final_hd.mp4"
        # re-merge since enhance pipeline does not include audio
        self.merge_audio_video(HD_video_path, audio_path, final_output_path)
        return final_output_path

    def generate_HD_upscale(self, video_path, output_path="output/", temp_frames_path="frames/"):
        # Split video into frames
        os.makedirs(temp_frames_path, exist_ok=True)
        vidcap = cv2.VideoCapture(video_path)
        numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print(f"FPS: {fps}, Frames: {numberOfFrames}")

        for frameNumber in tqdm(range(numberOfFrames)):
            success, image = vidcap.read()
            if not success:
                break
            frame_path = os.path.join(temp_frames_path, f"{frameNumber:04d}.jpg")
            cv2.imwrite(frame_path, image)
        
        # Run the upscaler on the frames
        wav2lip_hd_repo = "/models/wav2lip-HD/GFPGAN-master/"
        command = [
            "python", os.path.join(wav2lip_hd_repo, "inference_gfpgan.py"),
            "-i", temp_frames_path,
            "-o", output_path,
            "-v", "1.3",
            "-s", "1",
            "--only_center_face",
            "--bg_upsampler", "None",
            "--bg_tile", "0",
            "--upscale", "1"
        ]
        subprocess.run(command, check=True)

        # Combine frames back into a video
        restored_frames_path = os.path.join(output_path, "restored_imgs")
        dir_list = sorted(os.listdir(restored_frames_path))
        
        img_array = []
        for filename in dir_list:
            img = cv2.imread(os.path.join(restored_frames_path, filename))
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        
        out = cv2.VideoWriter(os.path.join(output_path, "output_hd.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for img in img_array:
            out.write(img)
        out.release()
        
        print("HD lip-sync video generation complete.")
        return os.path.join(output_path, "output_hd.mp4")


    def generate_real3d_lipsync(self, image_path, audio_path, output_path="/workspaces/ai-worker/runner/output/"):
        # Construct PYTHONPATH
        real3dportrait_path = "/models/Real3DPortrait"
        pythonpath = f"{real3dportrait_path}:{os.environ.get('PYTHONPATH', '')}"

        # wav16k_name = drv_aud[:-4] + '_16k.wav'
        # extract_wav_cmd = f"ffmpeg -i {drv_aud} -f wav -ar 16000 -v quiet -y {wav16k_name} -y"
        # subprocess.run(extract_wav_cmd, shell=True, check=True)
        # print(f"Extracted wav file (16khz) from {drv_aud} to {wav16k_name}.")

        # Define paths for Real3DPortrait
        os.environ['PYTHONPATH'] = './'
        output_video_path = os.path.join(output_path, "result.mp4")
        # Change to the desired directory

        os.chdir(real3dportrait_path)
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        

        # Run Real3DPortrait inference
        command = [
            "python", os.path.join(real3dportrait_path, "inference/real3d_infer.py"),
            "--src_img", image_path,
            "--drv_aud", os.path.join("/workspaces/ai-worker/runner", audio_path),
            "--out_name", output_video_path,
            "--out_mode", "final"
        ]

        # Prepend the PYTHONPATH to the command
        full_command = f"PYTHONPATH={pythonpath} " + " ".join(command)

        print(f"Running command: {full_command}")
        subprocess.run(full_command, shell=True, check=True)
        
        # Check if the output video was created
        if not os.path.exists(output_video_path):
            raise FileNotFoundError(f"Cannot find the output video file: {output_video_path}")
        
        print("Lip-sync video generation complete.")
        os.chdir("/workspaces/ai-worker/runner")
        return output_video_path

    def generate_wav2lip_lipsync(self, image_path, audio_path):
        # Construct PYTHONPATH
        wav2lip_path = "/models/wav2lip-HD"
        pythonpath = f"{wav2lip_path}:{os.environ.get('PYTHONPATH', '')}"
        wav2lip_folder_name = 'Wav2Lip-master'
        output_path = "output/"
        wav2lip_path = os.path.join(wav2lip_path, wav2lip_folder_name)
        lip_synced_output_path = os.path.join(output_path, "result.mp4")
        unProcessedFramesFolderPath = os.path.join(output_path, 'frames')
        temp_dir = "temp"
        temp_result_path = os.path.join(temp_dir, "result.avi")
        
        # Ensure output and temp directories exist
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Run Wav2Lip inference
        command = [
            "python", os.path.join(wav2lip_path, "inference.py"),
            "--checkpoint_path", os.path.join(wav2lip_path, "checkpoints", "wav2lip.pth"),
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", lip_synced_output_path
        ]
        # Prepend the PYTHONPATH to the command
        full_command = f"PYTHONPATH={pythonpath} " + " ".join(command)

        print(f"Running command: {full_command}")
        subprocess.run(full_command, shell=True, check=True)
        
        # Ensure unprocessed frames directory exists
        os.makedirs(unProcessedFramesFolderPath, exist_ok=True)
        
        # Check if the output video was created
        self.check_file(lip_synced_output_path)
        
        # Check if the temp result file was created
        self.check_file(temp_result_path)
        
        # Extract frames from the generated lip-synced video
        vidcap = cv2.VideoCapture(lip_synced_output_path)
        if not vidcap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {lip_synced_output_path}")
        
        numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("FPS: ", fps, "Frames: ", numberOfFrames)

        for frameNumber in tqdm(range(numberOfFrames)):
            success, image = vidcap.read()
            if not success:
                break
            cv2.imwrite(os.path.join(unProcessedFramesFolderPath, f"{frameNumber:04d}.jpg"), image)
        
        # Combine frames back into a video
        dir_list = sorted(os.listdir(unProcessedFramesFolderPath))
        img_array = []
        size = None
        for filename in dir_list:
            img = cv2.imread(os.path.join(unProcessedFramesFolderPath, filename))
            if img is None:
                continue
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        
        if size is None:
            raise ValueError("No valid frames found to determine video size.")
        
        out = cv2.VideoWriter(os.path.join(output_path, "output.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for img in img_array:
            out.write(img)
        out.release()
        
        print("Lip-sync video generation complete.")
        return os.path.join(output_path, "output.mp4")
    
    def check_file(self, file_path):
        if os.path.exists(file_path):
            print(f"File exists: {file_path}")
            print(f"File size: {os.path.getsize(file_path)} bytes")
            print(f"File permissions: {oct(os.stat(file_path).st_mode)[-3:]}")
        else:
            print(f"File does not exist: {file_path}")

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
        # Check if input files exist
        self.check_file(video_path)
        self.check_file(audio_path)

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