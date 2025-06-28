FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y git build-essential libgl1 python3-opencv \
 && pip install --upgrade pip \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && pip install git+https://github.com/facebookresearch/segment-anything.git

EXPOSE 8080

CMD ["python", "sam_api.py"]
