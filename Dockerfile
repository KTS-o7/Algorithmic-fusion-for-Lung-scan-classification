FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
#FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . /app

VOLUME /home/krishnatejaswis/Files/VSCode/Algorithmic-fusion-for-Lung-scan-classification

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
