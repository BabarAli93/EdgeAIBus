FROM ubuntu:22.04

WORKDIR /root

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git  \
    libgl1-mesa-glx \
    libglib2.0-0

# Optionally, set the default Python version to Python 3
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	
RUN pip install ultralytics

RUN pip install Flask

RUN pip install kubernetes

RUN mkdir yolov8

WORKDIR /root/yolov8

COPY *.pt .

# Copy Flask application files
COPY app8.py /root/app.py

# Expose port
EXPOSE 5000

# Set the entry point
CMD ["python", "/root/app.py"]
