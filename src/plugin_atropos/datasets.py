import math
import queue
import threading
import time
from functools import partial
from typing import List, Tuple, Callable, Optional, Iterator

import numpy as np
import torch
from datasets import IterableDataset
import requests
from datasets.iterable_dataset import ExamplesIterable
from tenacity import retry, stop_after_attempt, wait_exponential


class RemoteDataProvider:
    """
    Data provider that fetches from a remote API when needed.
    """
    def __init__(
            self,
            api_fetch_func: Callable,
            queue_threshold: int = 10,
            max_queue_size: int = 50,
            ttl: Optional[float] = None,
            fetch_delay: float = 0.1,
            worker_timeout: float = 1.0,
            pad_token_id: int = -1,
    ):
        """
        Args:
            api_fetch_func: Callable that returns data from the API
            queue_threshold: Minimum queue size before fetching more data
            max_queue_size: Maximum queue size to prevent unbounded growth
            ttl: Time-to-live for blocking on the queue when getting data
            fetch_delay: Delay between API fetch attempts in the worker
            worker_timeout: Timeout for worker thread operations
        """
        self.api_fetch_func = api_fetch_func
        self.queue_threshold = queue_threshold
        self.max_queue_size = max_queue_size
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.ttl = ttl
        self.fetch_delay = fetch_delay
        self.worker_timeout = worker_timeout
        self._example_counter = 0
        self.pad_token_id = pad_token_id

        # Start the worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        """
        Worker thread that fetches data from the API when queue is below threshold.
        """
        while not self.stop_event.is_set():
            try:
                # Check if we need to fetch more data
                if self.data_queue.qsize() < self.queue_threshold:
                    # print(f"Fetching data from API... q: {self.data_queue.qsize()}/{self.max_queue_size}")
                    # print(f"queue_threshold: {self.queue_threshold}")
                    try:
                        # Fetch data from the API
                        data = self.api_fetch_func()

                        # print(f"data len: {len(data)}")
                        for batch in data:
                            # print(f"batch len: {len(batch[0])}")
                            for prompt_ids, prompt_mask, completion_ids, completion_mask, advantages in zip(batch[0], batch[1], batch[2], batch[3], batch[4]):
                            #     sample_id = torch.Tensor([uuid.uuid4().int])
                            #     for input_id, label, score in zip(input_ids, labels, scores):
                            #         row = {"id": sample_id, "input_ids": input_id, "labels": label, "scores": score}
                            #         self.data_queue.put(row, timeout=self.worker_timeout)
                            #     print(prompt_ids.shape)
                            #     print(completion_ids.shape)
                                row = {"prompt_ids": prompt_ids, "prompt_mask": prompt_mask, "completion_ids": completion_ids, "completion_mask": completion_mask, "advantages": advantages}
                                # print("row: ")
                                # print(row)
                                self.data_queue.put(row, timeout=self.worker_timeout)

                    except Exception as e:
                        # Log or handle API fetch errors appropriately
                        if "object is not subscriptable" not in str(e):
                            print(f"API fetch error: {e}")
                        time.sleep(self.fetch_delay)
                        continue

                # Sleep before checking again
                time.sleep(self.fetch_delay)

            except queue.Full:
                # Queue is full, wait before trying again
                time.sleep(self.fetch_delay)
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(self.fetch_delay)

    def stop(self):
        """
        Method to signal that no more data will be added and iteration should stop
        when the queue is empty.
        """
        self.stop_event.set()
        # Wait for worker thread to finish
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

    def generate_examples_fn(self, **kwargs) -> Iterator[Tuple[int, dict]]:
        """
        Generator function that yields (key, example) tuples as expected by datasets.ExamplesIterable.
        This is the callable that ExamplesIterable expects.
        """
        while True:
            if self.stop_event.is_set() and self.data_queue.empty():
                # If stop has been signaled and the queue is empty, end the iteration
                break

            try:
                # Attempt to get data from the queue with blocking up to the TTL
                data = self.data_queue.get(timeout=self.ttl)
                # print("received data from queue: ", data)

                # Yield as (key, example) tuple
                # If data is a dict, use as-is; otherwise, wrap it
                if isinstance(data, dict):
                    yield self._example_counter, data
                else:
                    yield self._example_counter, {"data": data}

                self._example_counter += 1

            except queue.Empty:
                # If stop has been signaled but there's still data in the queue, check again
                if self.stop_event.is_set():
                    if not self.data_queue.empty():
                        # Try again without timeout to get remaining data
                        try:
                            data = self.data_queue.get_nowait()
                            if isinstance(data, dict):
                                yield self._example_counter, data
                            else:
                                yield self._example_counter, {"data": data}
                            self._example_counter += 1
                        except queue.Empty:
                            break
                    else:
                        break
                else:
                    # No stop signal yet, but no data available within TTL
                    # Sleep a bit and try again
                    time.sleep(self.fetch_delay)
                    continue

    def __del__(self):
        """
        Cleanup method to ensure the worker thread is stopped properly.
        """
        self.stop()


class RemoteIterableDataset(IterableDataset):
    """
    Wrapper class that creates a PyTorch IterableDataset from a remote API data source.
    Compatible with HuggingFace datasets library.
    """
    def __init__(
            self,
            api_fetch_func: Callable,
            queue_threshold: int = 10,
            max_queue_size: int = 50,
            ttl: Optional[float] = None,
            fetch_delay: float = 0.1,
            worker_timeout: float = 1.0,
    ):
        """
        Args:
            api_fetch_func: Callable that returns data from the API
            queue_threshold: Minimum queue size before fetching more data
            max_queue_size: Maximum queue size to prevent unbounded growth
            ttl: Time-to-live for blocking on the queue when getting data
            fetch_delay: Delay between API fetch attempts in the worker
            worker_timeout: Timeout for worker thread operations
        """
        # Create the data provider
        self.data_provider = RemoteDataProvider(
            api_fetch_func=api_fetch_func,
            queue_threshold=queue_threshold,
            max_queue_size=max_queue_size,
            ttl=ttl,
            fetch_delay=fetch_delay,
            worker_timeout=worker_timeout,
        )

        # Create the ExamplesIterable with the generate_examples_fn
        examples_iterable = ExamplesIterable(
            self.data_provider.generate_examples_fn, {},
        )

        super().__init__(self, examples_iterable)

    def stop(self):
        """Stop the worker thread and iteration."""
        self.data_provider.stop()

    def __iter__(self):
        """Forward iteration to the underlying dataset."""
        return iter(self.dataset)

    def __next__(self):
        """Forward next() call to the underlying dataset."""
        return next(iter(self.dataset))

    def __del__(self):
        """Cleanup when the object is deleted."""
        self.stop()

# Helper function for calculating padded length to a multiple
def _calculate_padded_length(batch_max_len: int, multiple: int) -> int:
    if batch_max_len == 0: # If max length is 0, padded length is also 0
        return 0
    return math.ceil(batch_max_len / multiple) * multiple

def pad_data_to_good_offset(data, batch_size: int, pad_token_id: int):
    good_multiple = 64

    all_prompts_np = []
    all_completions_np = []
    all_advantages = []

    # --- Stage 1: Process all items, split into prompts/completions, and collect ---
    for item_idx, item in enumerate(data["batch"]):
        # --- Score Processing ---
        scores = np.array(item["scores"], dtype=np.float32)
        if len(scores) > 1:
            scores_mean = scores.mean()
            scores_std = scores.std()
            scores = scores - scores_mean
            scores = scores / max(scores_std, 1e-8)

        if item["overrides"] is not None:
            for i in range(len(item["overrides"])):
                if item["overrides"][i].get("set_advantage_to_zero", False) and i < len(scores):
                    scores[i] = 0.0
        processed_scores = scores

        if not item["tokens"]:
            # If an item has no tokens, but has scores, we might need to handle this.
            # For now, we assume if no tokens, no corresponding advantages are added.
            # Or, if scores are per-item, not per-token-sequence, this needs different handling.
            # Assuming scores correspond to token sequences here.
            if len(item["tokens"]) != len(processed_scores):
                print(f"Warning: Item {item_idx} has {len(item['tokens'])} token sequences but {len(processed_scores)} scores. Mismatch.")
            continue

        prompt_length = 0
        if len(item["tokens"]) > 1:
            reference_tokens = item["tokens"][0]
            min_shared_len_for_item = len(reference_tokens)
            for i in range(1, len(item["tokens"])):
                current_seq_tokens = item["tokens"][i]
                len_to_compare = min(min_shared_len_for_item, len(current_seq_tokens))
                current_shared_count = 0
                for j in range(len_to_compare):
                    if reference_tokens[j] == current_seq_tokens[j]:
                        current_shared_count += 1
                    else:
                        break
                min_shared_len_for_item = min(min_shared_len_for_item, current_shared_count)
                if min_shared_len_for_item == 0:
                    break
            prompt_length = min_shared_len_for_item
        elif len(item["tokens"]) == 1: # Single sequence, prompt_length is 0
            prompt_length = 0


        for i in range(len(item["tokens"])):
            full_token_seq = item["tokens"][i]
            prompt_part = full_token_seq[:prompt_length]
            completion_part = full_token_seq[prompt_length:]

            all_prompts_np.append(np.array(prompt_part, dtype=np.int32))
            all_completions_np.append(np.array(completion_part, dtype=np.int32))
            if i < len(processed_scores):
                all_advantages.append(processed_scores[i])
            else:
                # This case should ideally not happen if scores align with tokens. Add a default if it does.
                all_advantages.append(0.0)
                print(f"Warning: Missing score for token sequence {i} in item {item_idx}. Appending 0.0 advantage.")


    # --- Stage 2: Determine global max lengths for padding ---
    if not all_prompts_np: # No data to process
        return [], [], [], [], []

    global_actual_max_prompt_len = max(len(p) for p in all_prompts_np) if all_prompts_np else 0
    global_padded_prompt_target_len = _calculate_padded_length(global_actual_max_prompt_len, good_multiple)

    global_actual_max_completion_len = max(len(c) for c in all_completions_np) if all_completions_np else 0
    global_padded_completion_target_len = _calculate_padded_length(global_actual_max_completion_len, good_multiple)

    # --- Stage 3: Batching and Padding to global target lengths ---
    prompt_ids_batches = []
    prompt_mask_batches = []
    completion_ids_batches = []
    completion_mask_batches = []
    advantages_batches = []

    num_samples = len(all_prompts_np)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    for i in range(0, num_samples, batch_size): # Iterate to form batches
        start_idx = i
        end_idx = min(i + batch_size, num_samples) # Handle last batch potentially being smaller

        if start_idx == end_idx: # Should not happen if num_samples > 0
            continue

        current_batch_prompts_np = all_prompts_np[start_idx:end_idx]
        current_batch_completions_np = all_completions_np[start_idx:end_idx]
        current_batch_advantages = np.array(all_advantages[start_idx:end_idx], dtype=np.float32)

        batch_padded_prompts_list = []
        batch_prompt_masks_list = []
        for prompt_arr in current_batch_prompts_np:
            pad_len = max(0, global_padded_prompt_target_len - len(prompt_arr))
            padded_p = np.concatenate([prompt_arr, np.full(pad_len, pad_token_id, dtype=np.int32)])
            mask_p = np.concatenate([np.ones(len(prompt_arr), dtype=np.int32), np.zeros(pad_len, dtype=np.int32)])
            batch_padded_prompts_list.append(padded_p)
            batch_prompt_masks_list.append(mask_p)

        prompt_ids_batches.append(torch.tensor(np.stack(batch_padded_prompts_list), dtype=torch.long))
        prompt_mask_batches.append(torch.tensor(np.stack(batch_prompt_masks_list), dtype=torch.long))

        batch_padded_completions_list = []
        batch_completion_masks_list = []
        for comp_arr in current_batch_completions_np:
            pad_len = max(0, global_padded_completion_target_len - len(comp_arr))
            padded_c = np.concatenate([comp_arr, np.full(pad_len, pad_token_id, dtype=np.int32)])
            mask_c = np.concatenate([np.ones(len(comp_arr), dtype=np.int32), np.zeros(pad_len, dtype=np.int32)])
            batch_padded_completions_list.append(padded_c)
            batch_completion_masks_list.append(mask_c)

        completion_ids_batches.append(torch.tensor(np.stack(batch_padded_completions_list), dtype=torch.long))
        completion_mask_batches.append(torch.tensor(np.stack(batch_completion_masks_list), dtype=torch.long))

        advantages_batches.append(torch.tensor(current_batch_advantages, dtype=torch.float32).view(-1, 1))

    return prompt_ids_batches, prompt_mask_batches, completion_ids_batches, completion_mask_batches, advantages_batches

def get_dataset(cfg, pad_token_id=None):
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def get_batch():
        data = requests.get(f"{cfg.atropos_server_host}:{cfg.atropos_server_port}/batch", timeout=10)
        try:
            res = data.json()
            return res
        except Exception as e:
            # print(e)
            # print(data)
            # print(data.content)
            return None

    def get_data(
            batch_size: int, seq_len: int
    ) -> List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
        """
        getting data from the api
        """
        batches = []
        while True:
            data = get_batch()
            if data["batch"] is not None:
                # In case the inference runs ahead of the training, we loop until we don't have any more data
                batches.append(pad_data_to_good_offset(data, batch_size, pad_token_id))
                return batches
            elif len(batches) > 0:
                # Return the batches
                return batches
            else:
                time.sleep(1)

    data_provider = RemoteDataProvider(
        api_fetch_func=partial(get_data, cfg.trl.num_generations, cfg.sequence_len),
        queue_threshold=5,
        max_queue_size=cfg.trl.num_generations * cfg.micro_batch_size * cfg.gradient_accumulation_steps * 20,
        ttl=1.0,
        fetch_delay=0.5,
        pad_token_id=pad_token_id,
    )

    dataset = IterableDataset(ExamplesIterable(data_provider.generate_examples_fn, {}))

    return dataset
