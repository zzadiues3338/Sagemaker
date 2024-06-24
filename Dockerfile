# Use the official PyTorch image from Docker Hub
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set the working directory
WORKDIR /opt/ml

# Copy the model and code into the container
COPY model.pth /opt/ml/model/
COPY inference.py /opt/ml/code/

# Install any additional Python packages
RUN pip install --no-cache-dir segmentation-models-pytorch
RUN pip install --no-cache-dir boto3

# Define environment variables
ENV PYTHONUNBUFFERED=TRUE

# Set the entry point for the container
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]
