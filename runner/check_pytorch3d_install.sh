#!/bin/bash

# Check if pytorch3d is needed
need_pytorch3d=$(python3 - <<'END'
try:
    import pytorch3d
    print("False")
except ModuleNotFoundError:
    print("True")
END
)


# Get PyTorch version and platform information
pytorch_version=$(python3 -c "import torch; print(torch.__version__)")
python_version_minor=$(python3 -c "import sys; print(sys.version_info.minor)")
cuda_version=$(python3 -c "import torch; print(torch.version.cuda)")

pytorch_version_str=$(echo "$pytorch_version" | cut -d'+' -f1 | tr -d '.')
version_str="py3${python_version_minor}_cu${cuda_version//./}_pyt${pytorch_version_str}"

pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${version_str}/download.html"

