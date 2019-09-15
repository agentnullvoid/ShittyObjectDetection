FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -yyq python python-pip python-opencv python-tk
RUN pip install --upgrade pip
RUN pip install numpy tensorflow matplotlib pillow

RUN mkdir /root/picturedump
RUN mkdir /ShittyDetection
ADD . /ShittyDetection

CMD ["python", "/ShittyDetection/detection.py"]
