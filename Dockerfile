FROM mambaorg/micromamba:1.5.1

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR /app
COPY . .

# Run as module to ensure 'src' is in path
CMD ["python", "-m", "src.trading.run_bot"]
