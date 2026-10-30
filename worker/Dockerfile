FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake SAM model weights into the image so cold starts don't hit Hugging Face
ENV HF_HOME=/models
RUN python -c "from transformers import SamProcessor, SamModel; \
SamProcessor.from_pretrained('facebook/sam-vit-large'); \
SamModel.from_pretrained('facebook/sam-vit-large')"
ENV HF_HUB_OFFLINE=1

COPY *.py .
CMD ["python", "main.py"]
