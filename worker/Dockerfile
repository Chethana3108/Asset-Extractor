FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake Grounding DINO and SAM ViT-Large models into image for fast cold starts
ENV HF_HOME=/models
RUN python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamProcessor, SamModel; \
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base'); \
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base'); \
SamProcessor.from_pretrained('facebook/sam-vit-large'); \
SamModel.from_pretrained('facebook/sam-vit-large')"
ENV HF_HUB_OFFLINE=1

COPY *.py .
CMD ["python", "main.py"]
