FROM continuumio/anaconda3:latest

WORKDIR /root

RUN mkdir model

RUN conda env create --file environment.yml
RUN conda activate serving

COPY ../../efficient-nlp ./

WORKDIR /root/efficient-nlp

RUN bash setup.sh

