# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the project files to the container
COPY . /app

# Install the package
RUN pip install .

# Set the entrypoint for the container
ENTRYPOINT ["python", "-m", "adappt_trainer.cli"]

# Default command if no arguments are provided
CMD ["--help"]
