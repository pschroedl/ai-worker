#!/bin/bash
cd /models/Real3DPortrait/
# Create the required directories
mkdir -p deep_3drecon/BFM
mkdir -p checkpoints/240126_real3dportrait_orig/audio2secc_vae
mkdir -p checkpoints/240126_real3dportrait_orig/secc2plane_torso_orig
mkdir -p checkpoints/pretrained_ckpts

echo "Directory structure created."

# Download 3DMM BFM Model
echo "Downloading 3DMM BFM Model..."
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

# Download Pre-trained Real3D-Portrait models
echo "Downloading Pre-trained Real3D-Portrait models..."
cd checkpoints
gdown https://drive.google.com/uc?id=1gz8A6xestHp__GbZT5qozb43YaybRJhZ
gdown https://drive.google.com/uc?id=1gSUIw2AkkKnlLJnNfS2FCqtaVw9tw3QF
unzip 240210_real3dportrait_orig.zip
unzip pretrained_ckpts.zip -x "__MACOSX/*" #maybe unecessary but we dont 
ls
cd ..

echo "All files downloaded and placed in the correct directories."