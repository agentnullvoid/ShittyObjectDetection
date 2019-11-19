FROM ubuntu:16.04

RUN apt-get update && \
    apt-get -y install sudo

ENV SUDO_USER desired_user_in_host

RUN useradd -m -d /home/${SUDO_USER} ${SUDO_USER} && \
    chown -R ${SUDO_USER} /home/${SUDO_USER} && \
    adduser ${SUDO_USER} sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ${SUDO_USER}
WORKDIR /home/${SUDO_USER}

RUN sudo apt-get update && \
    sudo apt-get install -yyq python python-pip python-opencv python-tk
RUN sudo pip install --upgrade pip

RUN mkdir picturedump
RUN mkdir ShittyObjectDetection
ADD . ShittyObjectDetection

RUN sudo pip install -r ShittyObjectDetection/requirements.txt

ENTRYPOINT ["sudo", "python", "ShittyObjectDetection/detection.py", "desired_user_in_host"]
