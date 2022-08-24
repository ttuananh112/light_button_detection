FROM python:3.7.13

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . /workspace/api

# install torch
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
# install torchvision
RUN pip install http://download.pytorch.org/whl/cu113/torchvision-0.13.1%2Bcu113-cp37-cp37m-linux_x86_64.whl

# install dependencies
RUN pip install -r /workspace/api/requirements.txt
