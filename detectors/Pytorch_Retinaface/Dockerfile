FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /root

COPY . .

RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
RUN apt-get install -y curl  python3.7-dev python3.7-distutils python3-pip
RUN python3 -m pip install virtualenv

RUN DEBIAN_FRONTEND=noninteractive  apt-get --yes install build-essential checkinstall
RUN DEBIAN_FRONTEND=noninteractive  apt-get --yes install libreadline-gplv2-dev \
    libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev

RUN virtualenv venv --python=python3.7
RUN . venv/bin/activate && \
    pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 scipy tqdm opencv-python

RUN chmod +x ./entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]