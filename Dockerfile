FROM mcr.microsoft.com/azureml/pytorch-1.7-ubuntu18.04-py37-cpu-inference:latest

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt
USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip uninstall opencv-contrib-python opencv-python opencv-contrib-python-headless
RUN pip install opencv-contrib-python
COPY ./app /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
