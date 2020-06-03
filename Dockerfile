FROM jupyter/datascience-notebook:latest

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
## datascience-notebook
########################################################################

USER root

RUN apt-get update
RUN apt-get install -y sqlite3 git emacs
RUN apt-get install -y sudo
RUN apt-get install -y aptitude ffmpeg imagemagick

# setup sudoers

RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo 'jovyan ALL=(ALL) NOPASSWD:ALL'  >> /etc/sudoers

# RUN conda update -c conda-forge jupyterlab

ADD requirements.txt /tmp/requirements.txt
RUN chmod 777 /tmp/requirements.txt
ADD lab /tmp/lab

# Start the notebook server
WORKDIR /tmp
RUN pip uninstall tensorflow
RUN conda config --set channel_priority false
RUN conda install tensorflow-gpu
RUN conda install -c conda-forge opencv
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/jupyterlab/jupyterlab.git

RUN mv /opt/conda/share/jupyter/lab /opt/conda/share/jupyter/lab.orig
RUN mv lab /opt/conda/share/jupyter/lab
RUN chmod -R 777 /opt/conda/share/jupyter/lab

# USER $NB_UID

WORKDIR /home/jovyan
# # Expose the notebook port
# EXPOSE 8000
EXPOSE 8888

# # Start the notebook server
CMD jupyter notebook --no-browser --port 8888 --ip=*
