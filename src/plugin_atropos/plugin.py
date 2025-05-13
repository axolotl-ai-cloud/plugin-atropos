import requests
from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_tokenizer

from .datasets import get_dataset


class AtroposPlugin(BasePlugin):

    def get_input_args(self) -> str:
        return "plugin_atropos.AtroposArgs"

    def load_datasets(self, cfg: DictDefault, preprocess: bool) -> "TrainDatasetMeta":
        from axolotl.common.datasets import TrainDatasetMeta
        tokenizer = load_tokenizer(cfg)
        return TrainDatasetMeta(
            train_dataset=get_dataset(cfg, tokenizer.pad_token_id),
            total_num_steps=cfg.max_steps,
        )

    def get_trainer_cls(self, cfg: DictDefault):
        if cfg.rl == "grpo":
            from .trainers.grpo import AtroposGRPOTrainer
            return AtroposGRPOTrainer

    def post_trainer_create(self, cfg, trainer):
        # register trainer with server
        atropos_server_host = cfg.atropos_server_host
        if not atropos_server_host.startswith("http"):
            atropos_server_host = f"http://{cfg.atropos_server_host}"
        # capture any errors during registration
        register_data = {
            "wandb_group": cfg.wandb_run_group or 'default',
            "wandb_project": cfg.wandb_project,
            "batch_size": cfg.trl.num_generations * cfg.gradient_accumulation_steps * cfg.world_size,
            # "batch_size": cfg.trl.num_generations,
            "max_token_len": cfg.sequence_len,
            "starting_step": cfg.atropos_starting_step,
            "checkpoint_dir": cfg.output_dir,
            "save_checkpoint_interval": cfg.save_steps,
            "num_steps": cfg.max_steps,
        }
        # print(register_data)
        try:
            data = requests.post(
                f"{atropos_server_host}:{cfg.atropos_server_port}/register",
                json=register_data,
                timeout=10,
            ).json()
        except requests.exceptions.RequestException as e:
            print(f"Error registering trainer with server: {e}")
