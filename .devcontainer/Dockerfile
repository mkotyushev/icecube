# Base CUDA devel image
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspaces

# Apt-get installs
RUN \
    apt update && \
    apt-get -y install git unzip wget


# Install Miniconda
RUN \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b && \
    rm Miniconda3-py38_23.1.0-1-Linux-x86_64.sh

# # Clone repo
# RUN \
#     git clone

# # Unzip cifar
# RUN cd otfusion \
#     unzip cifar.zip

# Add paths to PATH and PYTHONPATH
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV PYTHONPATH=$PYTHONPATH:/workspaces/icecube3/graphnet/src
ENV PYTHONPATH=$PYTHONPATH:/workspaces/icecube3/otfusion
ENV PYTHONPATH=$PYTHONPATH:/workspaces/icecube3/otfusion/cifar

# Install Python packages: torch first due to some dependencies importing torch during install
RUN \
    conda create --name graphnet python=3.8 gcc_linux-64 gxx_linux-64 libgcc cudatoolkit=11.7 -c conda-forge -y
RUN conda activate graphnet && \
    pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.13+cu117 && \
    pip install -r requirements.txt