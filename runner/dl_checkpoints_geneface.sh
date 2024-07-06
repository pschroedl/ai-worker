#!/bin/bash

# Ensure gdown is installed
pip install gdown

# Create directories and download files
mkdir -p data/binary/videos/May
cd data/binary/videos/May
gdown https://drive.google.com/uc?id=16fNJz5MbOMqHYHxcK_nPP4EPBXWjugR0
cd ../../../..

cd deep_3drecon/BFM
gdown https://drive.google.com/uc?id=1SPM3IHsyNAaVMwqZZGV6QVaV7I2Hly0v
gdown https://drive.google.com/uc?id=1MSldX9UChKEb3AXLVTPzZQcsbGD4VmGF
gdown https://drive.google.com/uc?id=180ciTvm16peWrcpl4DOekT9eUQ-lJfMU
gdown https://drive.google.com/uc?id=1KX9MyGueFB3M-X0Ss152x_johyTXHTfU
gdown https://drive.google.com/uc?id=19-NyZn_I0_mkF-F5GPyFMwQJ_-WecZIL
gdown https://drive.google.com/uc?id=11ouQ7Wr2I-JKStp2Fd1afedmWeuifhof
gdown https://drive.google.com/uc?id=18ICIvQoKX-7feYWP61RbpppzDuYTptCq
gdown https://drive.google.com/uc?id=1VktuY46m0v_n_d4nvOupauJkK4LF6mHE
cd ../..

mkdir -p checkpoints
cd checkpoints
gdown https://drive.google.com/uc?id=1O5C1vK4yqguOhgRQ7kmYqa3-E8q5H_65
unzip motion2video_nerf.zip
mkdir audio2motion_vae
cd audio2motion_vae
gdown https://drive.google.com/uc?id=1Qg5V-1-IyEgAOxb2PbBjHpYkizuy6njf
gdown https://drive.google.com/uc?id=1bKY5rn3vcAkv-2m1mui0qr4Fs38jEmy-
cd ../..
