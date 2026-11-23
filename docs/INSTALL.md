# Install the beta

The supported worker platform is Linux x86_64 with Python 3.12, Torch 2.13.0 built for CUDA 13.0 and a CUDA 13.0-capable NVIDIA driver. The application needs NumPy, plus aiohttp for networking and huggingface-hub for Hub assets.

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install build
python tools/build_release.py
(cd dist && sha256sum -c SHA256SUMS)
pip install 'dist/vvla-0.1.0b2-py3-none-any.whl[serve,hub]'
python tools/install_backend.py pi05
vvla doctor --model pi05
~~~

The installer reads the selected model's version lock, installs the SDK, runtime and backend wheels, verifies the Torch CUDA build, and runs pip check. The default worker path is ~/.local/share/vvla/envs/MODEL/bin/python. Set VVLA_ENV_ROOT to change the root. VVLA_PI05_PYTHON, VVLA_OFT_PYTHON, VVLA_LINGBOT_PYTHON and the explicit worker_python argument select existing workers. Custom locations passed with --directory need one of those overrides. Application and workers should use the same VVLA beta version.

### Model dependencies

| Model | Install command and assets |
| --- | --- |
| π0.5 | Run the Pi0.5 command above. Supply checkpoint config, weights, matching normalization data and tokenizer. The G4 cuBLASLt table ships in the backend wheel. OpenPI H10 has a separate input and tokenizer contract. |
| OFT | Run python tools/install_backend.py oft. The installer obtains the pinned OpenVLA-OFT source, Transformers fork source and vLLM 0.32 native wheel. Supply the model checkpoint, statistics and heads. The --sdpa option installs an SDPA-only worker. |
| LingBot | Run python tools/install_backend.py lingbot --flash-attn-wheel /wheels/flash_attn-2.8.3-COMPATIBLE.whl. The wheel must match the device and Torch/CUDA build. Installation prepares the pinned LingBot inference source, registers it and installs the vLLM native target. Supply the checkpoint, matching Qwen processor, training YAML and robot normalization. |

The model environments require a C++ compiler, Python development headers, Ninja and libjpeg development headers. Native extensions compile at first use. The installer accepts --native-wheel to use a local vLLM 0.32 binary and --skip-sources to use existing assets. Operators, tokenizer access permissions and model weights depend on the selected deployment. For FlashAttention, build the matching 2.8.3 wheel in the Torch 2.13 / CUDA 13.0 development environment.

~~~bash
python tools/build_lingbot_flash_attn.py --sdist /sources/flash_attn-2.8.3.tar.gz \
    --cuda-home /usr/local/cuda --arch 80 --output /wheels/flash-attn
~~~

Use the tested SM80 build family for Ampere/Ada and --arch 120 for Blackwell. The upstream dependency and checkpoint terms apply. [Notices](../NOTICE)

### Checkpoint and Hub assets

~~~bash
vvla pull org/model --revision COMMIT
vvla serve --model oft --checkpoint hf://org/model@COMMIT --offline
~~~

A checkpoint can carry vvla.json. Relative paths resolve from that checkpoint. A Hub asset can reference a file or a directory and carry its own revision. One manifest can therefore describe a deployment shared by local and remote clients. The --offline option requires all referenced Hub snapshots to be cached in the application and worker's accessible Hub cache.

~~~json
{
  "schema": 1,
  "assets": {
    "qwen": "hf://org/qwen-processor@COMMIT",
    "norm_stats": "robotwin_norm.json",
    "training_config": "lingbotvla_cli.yaml"
  }
}
~~~

Replace org/model and COMMIT with real repositories and revisions. For LingBot, qwen/, robotwin_norm.json and lingbotvla_cli.yaml beside the weights are discovered automatically. For the published RoboTwin checkpoint, the worker locates the single checkpoints/*/hf_ckpt directory and the top-level training YAML. The installer supplies the upstream RoboTwin normalization. VVLA fetches only the matching Qwen processor files at their pinned revision.

~~~bash
vvla pull robbyant/lingbot-vla-v2-6b-robotwin --revision 0451855729ec904f970600e0aec8b84661423afe
vvla serve --model lingbot --checkpoint hf://robbyant/lingbot-vla-v2-6b-robotwin@0451855729ec904f970600e0aec8b84661423afe --offline
~~~

The prepared source supplies the default RoboTwin robot configuration. For OFT, source and transformers_source normally come from the installer. Any field can be set with --asset NAME=PATH, assets in a JSON configuration, or the SDK assets argument. Upstream sources and model repositories are trusted deployment inputs. [OpenPI checkpoint contract](OPENPI_CONTRACT.md)

### Docker

~~~bash
docker build --build-arg MODEL=pi05 -t vvla:0.1.0b2 .
docker run --rm --gpus all vvla:0.1.0b2 doctor --model pi05
docker run --rm --gpus all -p 127.0.0.1:8000:8000 -v "$PWD/models:/models:ro" \
    vvla:0.1.0b2 serve --model pi05 --checkpoint /models/pi05 --host 0.0.0.0
~~~

The build installs one isolated worker. Select MODEL=oft or MODEL=lingbot as needed. For LingBot, place its matching flash_attn-2.8.3 wheel in native-wheels/ before building. The base CUDA development image is pinned by digest. CUDA 13 runtime access requires the NVIDIA container runtime on the host. The default service binds to loopback. Use a protected deployment network for a remote robot.
