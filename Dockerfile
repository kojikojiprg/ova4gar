FROM jupyter/minimal-notebook:latest

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

########################################################################
## nvidia/cuda
########################################################################

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.2.89

ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-2 && \
        ln -s cuda-10.2 /usr/local/cuda && \
        rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"

########################################################################
## minimal-notebook
########################################################################

USER root

RUN apt-get update \
 && apt-get install -y sqlite3 git vi \
 && apt-get install -y sudo \
 && apt-get install -y aptitude ffmpeg imagemagick \
 && apt-get install --no-install-recommends -y fonts-ipaexfont libglib2.0-0 git gcc \
 # for opencv
 && apt-get install --no-install-recommends -y libsm-dev libxrender-dev libxext-dev \
 # for openpose
 && apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# setup sudoers
RUN echo 'Defaults visiblepw'             >> /etc/sudoers \
 && echo 'jovyan ALL=(ALL) NOPASSWD:ALL'  >> /etc/sudoers

RUN mkdir -p /opt/local/work \
 && chown -R ${NB_USER}. /opt/local/work

USER $NB_USER
WORKDIR /opt/local/work

RUN conda update -n base conda \
 && conda install --yes --channel conda-forge jupyter_contrib_nbextensions \
 && conda install -y jupyterlab \
 && conda install -y poetry==1.0.5 \
 && conda clean --all --yes

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
 && poetry install

RUN jupyter nbextension enable code_prettify/code_prettify
