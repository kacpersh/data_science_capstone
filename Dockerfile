FROM python:3.8.10 as base

RUN apt-get -y update
RUN apt-get -y install python-setuptools python-dev build-essential python3-pip ffmpeg libsm6 libxext6 libcairo2-dev libjpeg-dev libgif-dev

ADD . /data_science_capstone
WORKDIR /home/kacper_krasowiak/

RUN pip install --upgrade pip
COPY requirements.txt .
RUN cat requirements.txt | xargs -n 1 -L 1 pip install