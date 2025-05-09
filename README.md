# plugin-atropos

A plugin to train LLMs using [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) with [Atropos](https://github.com/NousResearch/atropos), a large-scale RL Gym.

# Installation

Install axolotl + vllm if not already installed
```bash
pip install axolotl[vllm]
```

```bash
git clone https://github.com/NousResearch/atropos.git
cd atropos
pip install -e .
cd -
```

Install this plugin
```bash
pip install -e .
```

### Usage

```bash
# start the vLLM server (can take a few minutes), this will block the session
CUDA_VISIBLE_DEVICES=0,1 axolotl vllm-serve examples/train.yaml
#  CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-4B --port 9001 --host 0.0.0.0 --tensor-parallel-size=2 --max-model-len 4096 --kv-cache-dtype fp8

# in a new terminal session
# start the API server in the background and redirect both stdout and stderr
run-api &> logs.txt &
# start the RL environment, this will block the session
python ../atropos/environments/gsm8k_server.py serve --slurm false
```

Start the trainer
```bash
CUDA_VISIBLE_DEVICES=2,3 axolotl train examples/train.yaml
```
