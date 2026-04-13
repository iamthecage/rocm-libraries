sudo apt update && sudo apt install -y
wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
sudo apt install ./amdgpu-install_7.2.1.70201-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm

sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms

sudo usermod -a -G video,render $LOGNAME

sudo tee --append /etc/ld.so.conf.d/rocm.conf <<EOF
/opt/rocm/lib
/opt/rocm/lib64
EOF
sudo ldconfig

# bashrc exports (add to ~/.bashrc)
cat >> ~/.bashrc << 'BASHRC'

# ROCm
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIP_HOME=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH="$ROCM_HOME/bin:$ROCM_HOME/hip/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_HOME/lib:$ROCM_HOME/hip/lib:$LD_LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="$ROCM_HOME/include:$ROCM_HOME/hip/include:$CPLUS_INCLUDE_PATH"
export CMAKE_PREFIX_PATH="$ROCM_HOME:$CMAKE_PREFIX_PATH"

# GPU target
export PYTORCH_ROCM_ARCH="gfx1201"
export ROCM_ARCH=gfx1201
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0

# PyTorch / Triton
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PYTORCH_TUNABLEOP_ENABLED="1"
export PYTORCH_TUNABLEOP_FILENAME="$HOME/.pytorch_tunableop_results.csv"
export PYTORCH_TUNABLEOP_TUNING_DURATION="short"
BASHRC

#reboot at this point

# apt packages
sudo apt install -y git build-essential curl wget pkg-config \
  python3 python3-dev python3-pip python3-venv \
  libssl-dev libffi-dev zlib1g-dev

# snap packages
sudo snap install code --classic

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # or restart shell

# create master venv (python 3.12 specifically)
uv venv ~/master --python 3.12
source ~/master/bin/activate

# add master venv auto-activate to bashrc (now that it exists)
echo 'source ~/master/bin/activate' >> ~/.bashrc

# tensile dependencies
uv pip install pyyaml msgpack joblib rich pytest invoke cmake numpy

# ROCm PyTorch stack (cp312, rocm 7.2.1) from AMD repo
uv pip install \
  torch==2.8.0+rocm7.2.1.lw.gitd733adb1 \
  torchvision==0.23.0+rocm7.2.1.git824e8c87 \
  torchaudio==2.8.0+rocm7.2.1.git6e1c7fe9 \
  triton==3.4.0+rocm7.2.1.git0cace8d2 \
  --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/

# generate SSH key for GitHub
ssh-keygen -t ed25519 -C "your-email@example.com"
cat ~/.ssh/id_ed25519.pub
# paste the output into GitHub → Settings → SSH keys

# clone repos
git clone git@github.com:ROCm/rocm-libraries.git


# post-reboot smoke test
rocm-smi                                    # should show gfx1201
rocminfo | grep gfx                         # should list gfx1201
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

