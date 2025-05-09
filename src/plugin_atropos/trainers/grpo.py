from typing import Union, Any

import datasets
import torch
from axolotl.core.trainers.mixins import SchedulerMixin
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker
from trl import GRPOTrainer


def atropos_reward_placeholder(*args, **kwargs):
    return 0.0

class AtroposGRPOTrainer(SchedulerMixin, GRPOTrainer):
    def __init__(self, *args, **kwargs):
        reward_funcs = kwargs.pop("reward_funcs", None)
        if not reward_funcs:
            reward_funcs = [atropos_reward_placeholder]
        kwargs["reward_funcs"] = reward_funcs
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        print("=" * 20 + "Inside _generate_and_score_completions" + "=" * 20)
        mode = "eval" if self.control.should_evaluate else "train"

        # Consolidate inputs into single tensors
        consolidated_inputs = {
            "prompt_ids": torch.stack([input_dict["prompt_ids"] for input_dict in inputs]),
            "prompt_mask": torch.stack([input_dict["prompt_mask"] for input_dict in inputs]),
            "completion_ids": torch.stack([input_dict["completion_ids"] for input_dict in inputs]),
            "completion_mask": torch.stack([input_dict["completion_mask"] for input_dict in inputs]),
            "advantages": torch.stack([input_dict["advantages"] for input_dict in inputs])
        }

        # Now you can use consolidated_inputs
        prompt_ids = consolidated_inputs["prompt_ids"].squeeze(0)
        prompt_mask = consolidated_inputs["prompt_mask"].squeeze(0)
        completion_ids = consolidated_inputs["completion_ids"].squeeze(0)
        completion_mask = consolidated_inputs["completion_mask"].squeeze(0)
        advantages = consolidated_inputs["advantages"].squeeze(0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        logits_to_keep = completion_ids.size(1)

        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                    )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        return self._generate_and_score_completions(accumulated_local_batch)
        # return accumulated_local_batch[0]

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        print(f"num_batches: {num_batches}")
        batch_samples = []
        num_items_in_batch = None

        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        return batch_samples, num_items_in_batch

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": (self._train_batch_size // self.num_generations) * self.args.gradient_accumulation_steps,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        # just return a SequentialSampler
        return torch.utils.data.SequentialSampler(self.train_dataset)