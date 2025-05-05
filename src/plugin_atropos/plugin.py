from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault


class AtroposPlugin(BasePlugin):

    def get_input_args(self) -> str:
        return "plugin_atropos.AtroposArgs"

    def load_datasets(self, cfg: DictDefault, preprocess: bool) -> "TrainDatasetMeta":
        pass

    def get_trainer_cls(self, cfg: DictDefault):
        if cfg.rl == "grpo":
            from .trainers.grpo import AtroposGRPOTrainer
            return AtroposGRPOTrainer

    def post_trainer_create(self, cfg, trainer):
        # register trainer with server
        pass
