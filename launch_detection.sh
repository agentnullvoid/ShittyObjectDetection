#!/bin/bash

docker run -it --device=/dev/video0:/dev/video0:rwm -v ~/test:/root/picturedump agentnullvoid/shittydetection:latest
