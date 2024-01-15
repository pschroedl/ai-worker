# Based on https://github.com/huggingface/api-inference-community/blob/main/docker_images/diffusers/Dockerfile

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04
LABEL maintainer="Yondon Fu <yondon@livepeer.org>"

# Add any system dependency here
# RUN apt-get update -y && apt-get install libXXX -y

ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites
RUN apt-get update && \
  apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
  xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

# Install pyenv
RUN curl https://pyenv.run | bash

# Set environment variables for pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install your desired Python version
ARG PYTHON_VERSION=3.11
RUN pyenv install $PYTHON_VERSION && \
  pyenv global $PYTHON_VERSION && \
  pyenv rehash

# Upgrade pip and install your desired packages
ARG PIP_VERSION=23.3.2
RUN pip install --no-cache-dir --upgrade pip==${PIP_VERSION} setuptools wheel && \
  pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

WORKDIR /app
COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Most DL models are quite large in terms of memory, using workers is a HUGE
# slowdown because of the fork and GIL with python.
# Using multiple pods seems like a better default strategy.
# Feel free to override if it does not make sense for your library.
ARG max_workers=1
ENV MAX_WORKERS=$max_workers
ENV HUGGINGFACE_HUB_CACHE=/models
ENV DIFFUSERS_CACHE=/models
ENV MODEL_DIR=/models

COPY app/ /app/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]