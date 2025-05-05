from axolotl.core.trainers.mixins import SchedulerMixin
from transformers import Trainer


class AtroposGRPOTrainer(SchedulerMixin, Trainer):
    pass