FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -yyq python python-pip python-opencv python-tk
RUN pip install --upgrade pip

RUN mkdir /root/picturedump
RUN mkdir /ShittyObjectDetection
ADD . /ShittyObjectDetection

RUN pip install -r /ShittyObjectDetection/requirements.txt

CMD ["python", "/ShittyObjectDetection/detection.py"]
