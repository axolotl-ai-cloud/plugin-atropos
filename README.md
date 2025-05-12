# plugin-atropos

A plugin to train LLMs using [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) with [Atropos](https://github.com/NousResearch/atropos), a large-scale RL Gym.

# Installation

Install axolotl + vllm if not already installed
```bash
pip install axolotl[vllm,flash-attn]
```

Install Atropos
```bash
git clone https://github.com/NousResearch/atropos.git
cd atropos
git checkout add-support-for-custom-api-servers
pip install -e .
cd -
```

Install this plugin
```bash
git clone https://github.com/axolotl-ai-cloud/plugin-atropos.git
cd plugin-atropos
pip install -e .
```

### Usage

Note: requires 3 GPUs: 0,1 runs vllm, 2 runs trainer

```bash
# start the vLLM server (can take a few minutes), this will block the session
CUDA_VISIBLE_DEVICES=0,1 axolotl vllm-serve examples/train-fft.yaml --serve-module plugin_atropos.vllm_serve
#  CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-4B --port 9001 --host 0.0.0.0 --tensor-parallel-size=2 --max-model-len 4096 --kv-cache-dtype fp8

# in a new terminal session
# start the API server in the background and redirect both stdout and stderr
run-api &> logs.txt &
# start the RL environment, this will block the session
python examples/gsm8k_server.py serve --slurm false
```

Start the trainer
```bash
CUDA_VISIBLE_DEVICES=2 axolotl train examples/train-fft.yaml
```
