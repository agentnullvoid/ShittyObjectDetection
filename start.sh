#!/bin/bash

docker run -it --device=/dev/video1:/dev/video0:rwm -v /home/desired_user_in_host/picturedump:/home/desired_user_in_host/picturedump agentnullvoid/shittydetection:latest
