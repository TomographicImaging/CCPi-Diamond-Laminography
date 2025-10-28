FROM mambaorg/micromamba:2.0.8-debian12-slim

COPY environment.yml /tmp/environment.yml

RUN micromamba env create -f /tmp/environment.yml &&     micromamba clean -a -y

WORKDIR /app

COPY scripts/ /app/scripts/
COPY alignment/ /app/alignment/
COPY artefacts/ /app/artefacts/
COPY pyproject.toml /app/

RUN micromamba run -n laminography_alignment pip install .

EXPOSE 8888

CMD ["micromamba", "run", "-n", "laminography_alignment", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

