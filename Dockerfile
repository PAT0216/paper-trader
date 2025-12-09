FROM continuumio/miniconda3

WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create Conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "paper-trader", "/bin/bash", "-c"]

COPY . .

# Run the main entry point
CMD ["conda", "run", "--no-capture-output", "-n", "paper-trader", "python", "main.py"]
