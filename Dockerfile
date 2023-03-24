FROM runpod/tensorflow
WORKDIR controlnet
ADD ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U xformers
RUN pip install colorgram.py
RUN pip uninstall -y nvidia_cublas_cu11

RUN apt-get update --allow-insecure-repositories && apt-get install -y libsm6 libxrender1 libfontconfig1
RUN apt-get install -y libsm6 libxext6

ADD ./training training
ADD ./models models
ADD . .

# Runpod requires template dockerfiles run forever 
ADD start.sh /

RUN chmod +x /start.sh

CMD [ "/start.sh" ]