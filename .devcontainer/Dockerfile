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

# Install torch & other related deps: 
# - torch first due to some dependencies importing torch during install
RUN conda init bash
RUN \
    conda create --name graphnet python=3.8 gcc_linux-64 gxx_linux-64 libgcc cudatoolkit=11.7 -c conda-forge -y
RUN /root/miniconda3/envs/graphnet/bin/python -m pip install --upgrade pip

RUN /root/miniconda3/envs/graphnet/bin/pip install --default-timeout=1000 --extra-index-url https://download.pytorch.org/whl/cu117 torch==2.0.0+cu117 torchvision==0.15.0+cu117
COPY ../requirements.txt .
RUN /root/miniconda3/envs/graphnet/bin/pip install --default-timeout=1000 -r requirements.txt 
RUN /root/miniconda3/envs/graphnet/bin/pip install matplotlib tables gpytorch tabulate seaborn mock

ENV PYTHONPATH=/workspace/icecube/lib/graphnet/src:$PYTHONPATH
ENV PYTHONPATH=/workspace/icecube/lib/otfusion:$PYTHONPATH
ENV PYTHONPATH=/workspace/icecube/lib/otfusion/cifar:$PYTHONPATH
ENV PYTHONPATH=/workspace/icecube/lib/loss-surface-simplexes:$PYTHONPATH
ENV PYTHONPATH=/workspace/icecube/lib/loss-surface-simplexes/simplex:$PYTHONPATH
ENV PYTHONPATH=/workspace/icecube/lib/loss-surface-simplexes/simplex/models:$PYTHONPATH
ENV PYTHONPATH=/workspace/icecube/lib:$PYTHONPATH
ENV PATH=/workspace/icecube/lib/graphnet/src:$PATH
ENV PATH=/workspace/icecube/lib/otfusion:$PATH
ENV PATH=/workspace/icecube/lib/otfusion/cifar:$PATH
ENV PATH=/workspace/icecube/lib/loss-surface-simplexes:$PATH
ENV PATH=/workspace/icecube/lib/loss-surface-simplexes/simplex:$PATH
ENV PATH=/workspace/icecube/lib/loss-surface-simplexes/simplex/models:$PATH
ENV PATH=/workspace/icecube/lib:$PATH
