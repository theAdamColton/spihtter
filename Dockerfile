# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
WORKDIR /code
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y git
RUN apt-get install -y cargo
RUN apt-get install -y fish
RUN pip install -r requirements.txt
RUN git clone https://github.com/Dao-AILab/causal-conv1d
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install ./causal-conv1d/
RUN git clone https://github.com/state-spaces/mamba
RUN MAMBA_FORCE_BUILD=TRUE pip install ./mamba/
COPY . spihtter
SHELL ["fish"]
