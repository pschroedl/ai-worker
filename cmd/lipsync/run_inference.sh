#!/bin/bash
# Check if the volume is mounted, then create the symlink
# if [ -d "/models" ]; then
#   ln -sf /models/models--yerfor--Real3DPortrait/deep_3drecon/BFM/01_MorphableModel.mat  /models--yerfor--Real3DPortrait/deep_3drecon/BFM/01_MorphableModel.mat
#   ln -sf  /models/models--yerfor--Real3DPortrait/deep_3drecon/BFM/Exp_Pca.bin /models--yerfor--Real3DPortrait/deep_3drecon/BFM/Exp_Pca.bin
#   ln -sf  /models/models--yerfor--Real3DPortrait/deep_3drecon/BFM/std_exp.txt /models--yerfor--Real3DPortrait/deep_3drecon/BFM/std_exp.txt
# fi

# Check if the volume is mounted, then create the symlink
if [ -d "/models" ]; then
  required_files=(
      "deep_3drecon/BFM/01_MorphableModel.mat"
      "deep_3drecon/BFM/BFM_exp_idx.mat"
      "deep_3drecon/BFM/BFM_front_idx.mat"
      "deep_3drecon/BFM/BFM_model_front.mat"
      "deep_3drecon/BFM/Exp_Pca.bin"
      "deep_3drecon/BFM/facemodel_info.mat"
      "deep_3drecon/BFM/std_exp.txt"
      "deep_3drecon/reconstructor_opt.pkl"
      "checkpoints/240210_real3dportrait_orig/audio2secc_vae/config.yaml"
      "checkpoints/240210_real3dportrait_orig/audio2secc_vae/model_ckpt_steps_400000.ckpt"
      "checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/config.yaml"
      "checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/model_ckpt_steps_100000.ckpt"
      "checkpoints/pretrained_ckpts/mit_b0.pth"
  )

  # Create symbolic links for each required file
  for file in "${required_files[@]}"; do
    ln -sf "/models/models--yerfor--Real3DPortrait/$file" "/models--yerfor--Real3DPortrait/$file"
  done

  echo "Symlinks created successfully."
else
  echo "Error: /models directory is not mounted."
fi

# Run the Real3DPortrait inference command
python /models--yerfor--Real3DPortrait/inference/real3d_infer.py
