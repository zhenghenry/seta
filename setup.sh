# 1. install miniforge
SCRIPT_PATH=$(realpath "$0")
PROJECT_DIR=$(dirname "$SCRIPT_PATH")
#!/bin/bash
cd $PROJECT_DIR/../
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
# run the installer without interactive mode, all yes to prompts
bash Miniforge3-$(uname)-$(uname -m).sh -b
# initialize conda
source ~/miniforge3/bin/activate
conda init
conda create -n terminal_agent python=3.12 -y
conda activate terminal_agent
pip install uv
# 2. install dependencies
# if nvcc is not found, install cuda toolkit
if ! command -v nvcc &> /dev/null
then
    echo "nvcc not found, installing CUDA toolkit..."
    mamba install nvidia::cuda-toolkit -y
else
    echo "nvcc found, skipping CUDA toolkit installation."
fi

cd ${PROJECT_DIR}/external/camel && uv pip install -e .
cd ${PROJECT_DIR}/external/terminal-bench && uv pip install -e .
cd ${PROJECT_DIR}/external/areal && uv pip install -e .[all]

uv pip install flash-attn==2.8.3
uv pip install -U datasets transformers
uv pip install "numpy<2.3,>=2.0"

# 3. install docker if not found
if ! command -v docker &> /dev/null
then
    echo "Docker not found, installing..."
    cd ${PROJECT_DIR}/../
    # install docker using the convenience script
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh ./get-docker.sh
    # add current user to docker group
    sudo usermod -aG docker $USER
    newgrp docker
else
    echo "Docker found, skipping installation."
fi


# 4. modify docker to increase network address pool

DOCKER_DAEMON_CONFIG='/etc/docker/daemon.json'

# Backup existing daemon.json if it exists
if [ -f "$DOCKER_DAEMON_CONFIG" ]; then
    echo "Backing up existing Docker daemon configuration..."
    sudo cp "$DOCKER_DAEMON_CONFIG" "${DOCKER_DAEMON_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Create or update daemon.json with network pool settings
echo "Configuring Docker daemon..."
sudo tee "$DOCKER_DAEMON_CONFIG" > /dev/null <<EOF
{
  "default-address-pools": [
    {
      "base": "10.200.0.0/16",
      "size": 24
    }
  ]
}
EOF

# Restart Docker to apply changes
echo "Restarting Docker daemon..."
sudo systemctl restart docker

echo "Docker configuration complete!"
