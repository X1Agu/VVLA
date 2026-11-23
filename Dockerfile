FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev build-essential git ninja-build \
    libjpeg-dev libgl1 libglib2.0-0 ca-certificates && rm -rf /var/lib/apt/lists/*
ENV VVLA_ENV_ROOT=/opt/vvla/envs PATH=/opt/vvla/app/bin:$PATH
WORKDIR /opt/vvla
COPY . .
ARG MODEL=pi05
# For LingBot copy the compatible flash_attn-2.8.3 wheel into native-wheels/ before building.
RUN python3.12 -m venv app && app/bin/pip install build==1.3.0 \
    && python tools/build_release.py && pip install 'dist/vvla-0.1.0b2-py3-none-any.whl[serve,hub]' \
    && if [ "$MODEL" = lingbot ]; then python tools/install_backend.py "$MODEL" --flash-attn-wheel native-wheels/flash_attn-2.8.3-*.whl; \
       else python tools/install_backend.py "$MODEL"; fi
EXPOSE 8000
ENTRYPOINT ["vvla"]
CMD ["--help"]
