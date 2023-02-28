# Base CUDA devel image
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspaces/icecube

# Apt-get installs
RUN \
    apt update && \
    apt-get -y install git unzip wget

# Install Miniconda
RUN \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b && \
    rm Miniconda3-py38_23.1.0-1-Linux-x86_64.sh

# Add paths to PATH and PYTHONPATH
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV PYTHONPATH=$PYTHONPATH:/workspaces/icecube/graphnet/src
ENV PYTHONPATH=$PYTHONPATH:/workspaces/icecube/otfusion
ENV PYTHONPATH=$PYTHONPATH:/workspaces/icecube/otfusion/cifar

# Install Python packages: torch first due to some dependencies importing torch during install
WORKDIR /workspaces/icecube
RUN \
    conda create --name graphnet python=3.8 gcc_linux-64 gxx_linux-64 libgcc cudatoolkit=11.7 -c conda-forge -y
RUN /root/miniconda3/bin/pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.13+cu117
RUN /root/miniconda3/bin/pip install \
        torch-cluster==1.6.0 \
        torch_scatter==2.0.9 \
        torch-sparse==0.6.13 \
        torch_geometric==2.0.4 \
        POT==0.8.2
# Workarnound for torchvision's pillow dependency resolving issue
RUN /root/miniconda3/bin/python -m pip install --upgrade pip
RUN /root/miniconda3/bin/python -m pip install --upgrade pillow
RUN /root/miniconda3/bin/pip install -i https://download.pytorch.org/whl/cu117 torchvision==0.14.0+cu117