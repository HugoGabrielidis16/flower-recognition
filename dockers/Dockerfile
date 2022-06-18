FROM python:3.8
RUN mkdir /test

ADD . /test

WORKDIR /test


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "test.py"]



