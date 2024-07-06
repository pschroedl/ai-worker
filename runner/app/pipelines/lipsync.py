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
import cv2
from tqdm import tqdm
from PIL import Image, ImageFile
from fastapi import Form
from fastapi.responses import JSONResponse
from typing import List, Tuple, Optional
from io import BytesIO
import base64
import tempfile

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

    def __call__(self, text, image_file, seed=None):
        # Step 1: Read the text file for transcription
        # transcription = self.read_text_file(text_file)
        
        # Step 2: Generate audio from transcription using TTS
        # audio_path = self.generate_speech(text)
        audio_path = "output_speech.wav"
        # Step 3: Generate video frames using Stable Diffusion
        # temp_video_path = self.generate_video(image_file)
        # dummy_video_path = "dummy_video.mp4"
        # self.generate_dummy_video(image_file, dummy_video_path, 25, 30)
        
        temp_image_file_path = save_image_to_temp_file(image_file)
        # Step 4: Generate lip-synced video using Wav2Lip
        # output_video_path = self.generate_lipsync(temp_image_file_path, audio_path)
        output_video_path = self.generate_real3d_lipsync(temp_image_file_path, audio_path)

        return output_video_path
    
    def generate_dummy_video(self, image_path, output_video_path="dummy_video.mp4", num_frames=25, fps=7):
    # Open the image
        image = Image.open(image_path)
        
        # Create the same frame multiple times
        video_frames = [image] * num_frames
        
        # Export the frames to a video
        export_to_video(video_frames, output_video_path, fps=fps)
        
        print(f"Dummy video created at {output_video_path}")

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
    


    def generate_geneface_lipsync(self, video_path, audio_path, output_path="output/", model="geneface++"):
        # Define paths for GeneFace++
        geneface_path = "/models/GeneFacePlusPlus"
        a2m_ckpt = os.path.join(geneface_path, "checkpoints/audio2motion_vae")
        head_ckpt = os.path.join(geneface_path, "checkpoints/motion2video_nerf/may_head")
        torso_ckpt = os.path.join(geneface_path, "checkpoints/motion2video_nerf/may_torso")
        lip_synced_output_path = os.path.join(output_path, "result.mp4")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
            
        # Construct PYTHONPATH
        pythonpath = f"{geneface_path}:{os.environ.get('PYTHONPATH', '')}"

        # Run GeneFace++ inference
        command = [
        "python", os.path.join(geneface_path, "inference/genefacepp_infer.py"),
        "--a2m_ckpt", a2m_ckpt,
        "--head_ckpt", head_ckpt,
        "--torso_ckpt", torso_ckpt,
        "--drv_aud", audio_path,
        "--out_name", lip_synced_output_path
        ]

        # Prepend the PYTHONPATH to the command
        full_command = f"PYTHONPATH={pythonpath} " + " ".join(command)

        print(f"Running command: {full_command}")
        subprocess.run(full_command, shell=True, check=True)
        
        # Check if the output video was created
        if not os.path.exists(lip_synced_output_path):
            raise FileNotFoundError(f"Cannot find the output video file: {lip_synced_output_path}")
        
        print("Lip-sync video generation complete.")
        return lip_synced_output_path


    def check_file(self, file_path):
        if os.path.exists(file_path):
            print(f"File exists: {file_path}")
            print(f"File size: {os.path.getsize(file_path)} bytes")
            print(f"File permissions: {oct(os.stat(file_path).st_mode)[-3:]}")
        else:
            print(f"File does not exist: {file_path}")



    def generate_real3d_lipsync(self, src_img, drv_aud, drv_pose=None, bg_img=None, output_path="/workspaces/ai-worker/runner/output/", out_name="output.mp4", out_mode="final", low_memory_usage=False):
        # Construct PYTHONPATH
        wav16k_name = drv_aud[:-4] + '_16k.wav'
        real3dportrait_path = "/models/Real3DPortrait"
        pythonpath = f"{real3dportrait_path}:{os.environ.get('PYTHONPATH', '')}"
        extract_wav_cmd = f"ffmpeg -i {drv_aud} -f wav -ar 16000 -v quiet -y {wav16k_name} -y"
        subprocess.run(extract_wav_cmd, shell=True, check=True)
        print(f"Extracted wav file (16khz) from {drv_aud} to {wav16k_name}.")

        # Define paths for Real3DPortrait
        os.environ['PYTHONPATH'] = './'
        output_video_path = os.path.join(output_path, out_name)
        # Change to the desired directory

        os.chdir(real3dportrait_path)
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        

        # Run Real3DPortrait inference
        command = [
            "python", os.path.join(real3dportrait_path, "inference/real3d_infer.py"),
            "--src_img", src_img,
            "--drv_aud", os.path.join("/workspaces/ai-worker/runner", wav16k_name),
            "--out_name", output_video_path,
            "--out_mode", out_mode
        ]
        
        # if drv_pose:
        #     command.extend(["--drv_pose", drv_pose])
        # if bg_img:
        #     command.extend(["--bg_img", bg_img])
        # if low_memory_usage:
        #     command.append("--low_memory_usage")

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

    def generate_lipsync(self, video_path, audio_path, output_path="output/", model="wav2lip"):
        # Define paths
        base_path = "/models/wav2lip-HD"
        wav2lip_folder_name = 'Wav2Lip-master'
        wav2lip_path = os.path.join(base_path, wav2lip_folder_name)
        lip_synced_output_path = os.path.join(output_path, "result.mp4")
        unProcessedFramesFolderPath = os.path.join(output_path, 'frames')
        temp_dir = "temp"
        temp_result_path = os.path.join(temp_dir, "result.avi")
        
        # Ensure output and temp directories exist
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Check if paths and files exist
        self.check_file(video_path)
        self.check_file(audio_path)
        self.check_file(os.path.join(wav2lip_path, "inference.py"))
        self.check_file(os.path.join(wav2lip_path, "checkpoints", f"{model}.pth"))
        
        # Run Wav2Lip inference
        command = [
            "python", os.path.join(wav2lip_path, "inference.py"),
            "--checkpoint_path", os.path.join(wav2lip_path, "checkpoints", f"{model}.pth"),
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", lip_synced_output_path
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        
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


    def generate_lipsync_from_image(self, image_path, audio_path, output_path="output/", model="wav2lip"):
        # Define paths
        base_path = "/models/wav2lip-HD"
        wav2lip_folder_name = 'Wav2Lip-master'
        wav2lip_path = os.path.join(base_path, wav2lip_folder_name)
        lip_synced_output_path = os.path.join(output_path, "result.mp4")
        unProcessedFramesFolderPath = os.path.join(output_path, 'frames')
        temp_dir = "temp"
        temp_result_path = os.path.join(temp_dir, "result.avi")
        
        # Ensure output and temp directories exist
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Check if paths and files exist
        self.check_file(image_path)
        self.check_file(audio_path)
        self.check_file(os.path.join(wav2lip_path, "inference.py"))
        self.check_file(os.path.join(wav2lip_path, "checkpoints", f"{model}.pth"))
        
        # Run Wav2Lip inference
        command = [
            "python", os.path.join(wav2lip_path, "inference.py"),
            "--checkpoint_path", os.path.join(wav2lip_path, "checkpoints", f"{model}.pth"),
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", lip_synced_output_path
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        
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

    def generate_video(self, image_file):
        # Load Stable Diffusion model for image-to-video generation
        # self.stable_video_model_id = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        # self.stable_video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        #     self.stable_video_model_id, cache_dir=get_model_dir(), variant="fp16", torch_dtype=torch.float16
        # ).to(self.device)
    
        self.stable_video_pipeline = self.load_stable_diffusion_model()
        self.stable_video_pipeline.en_and_decode_n_samples_a_time = (
                    8  # Decode n frames at a time
                )
        global lowvram_mode
        lowvram_mode = True

        image = Image.open(image_file)
        image = [image.resize((1024, 576))]
        image *= 2

        # Reduce batch size and number of frames
        num_frames = 10  # Reduce the number of frames to fit within GPU memory
        decode_chunk_size = 4  # Reduce decode chunk size
        
        video_frames = self.stable_video_pipeline(image=image, num_frames=num_frames, decode_chunk_size=decode_chunk_size).frames[0]
        
        # Unload the Stable Diffusion model from the GPU
        print("Unloading Stable Diffusion model")
        self.unload_model(self.stable_video_pipeline)
        video_path = "generated_video.mp4"
        export_to_video(video_frames, video_path, fps=7)
        return video_path

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

def image_to_data_url(image_file):
    # Convert SpooledTemporaryFile to PIL Image
    image = Image.open(image_file)
    
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def save_image_to_temp_file(image_file):
    image = Image.open(image_file)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file, format="JPEG")
    temp_file.close()
    return temp_file.name