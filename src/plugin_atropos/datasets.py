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
                                row = {"prompt_ids": prompt_ids, "prompt_mask": prompt_mask, "completion_ids": completion_ids, "completion_mask": completion_mask, "advantages": advantages}
                                # print("row: ")
                                # print(row)
                                self.data_queue.put(row, timeout=self.worker_timeout)

                    except Exception as e:
                        # Log or handle API fetch errors appropriately
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


def pad_data_to_good_offset(data, batch_size: int, pad_token_id: int):
    max_token_len = max(
        [max([len(x) for x in item["tokens"]]) for item in data["batch"]]
    )
    # usually 64 is a good choice to ensure nonweird scaling behavior on GPUS
    # so we pad to the nearest multiple of 64
    good_multiple = 64
    if (max_token_len - 1) % good_multiple != 0:
        max_token_len = math.ceil((max_token_len - 1) / good_multiple) * good_multiple
        token_setup_len = (
                max_token_len + 1
        )  # add 1 so we can make it causal at the proper length
    else:
        token_setup_len = max_token_len
        max_token_len = (
                max_token_len - 1
        )  # since it's causal we need to remove the last bit...
    # pad all tokens to max_token_len and add to lists
    prompt_ids = list()
    prompt_mask = list()
    completion_ids = list()
    completion_mask = list()
    advantages = list()
    for item in data["batch"]:
        scores = item["scores"]
        scores = np.array(scores)
        # check if we have more than 1 score...
        if len(scores) > 1:
            scores = scores - scores.mean()
            scores = scores / max(scores.std(), 1e-8)
        item["scores"] = scores
        if item["overrides"] is not None:
            for i in range(len(item["overrides"])):
                if item["overrides"][i].get("set_advantage_to_zero", False):
                    item["scores"][i] = 0
        # TODO for each of the rows in item["tokens"], find the shared prefix length and lets' split
        # the token into prompt_ids and completion_ids. We don't need pad the prompt_ids, and we
        # can pad the completion_ids based on the original combined length

        # Initialize new keys for prompt_ids and completion_ids
        item["prompt_ids"] = []
        item["prompt_mask"] = []
        item["completion_ids"] = []
        item["completion_mask"] = []

        # Find the shared prefix length across all rows in item["tokens"]
        # Get the first row as reference
        reference_tokens = item["tokens"][0]
        # Compare with each other row to find shared prefix
        prefix_lengths = []
        for i in range(1, len(item["tokens"])):
            # Find where tokens start to differ
            min_len = min(len(reference_tokens), len(item["tokens"][i]))
            shared_len = 0
            for j in range(min_len):
                if reference_tokens[j] == item["tokens"][i][j]:
                    shared_len += 1
                else:
                    break
            prefix_lengths.append(shared_len)
        # Use the minimum shared prefix length as our prompt length
        prompt_length = min(prefix_lengths) if prefix_lengths else 0

        # Split tokens into prompt_ids and completion_ids
        for i in range(len(item["tokens"])):
            # Split at the determined prompt length
            prompt = item["tokens"][i][:prompt_length]
            completion = item["tokens"][i][prompt_length:]

            # Store the split sequences in new keys
            item["prompt_ids"].append(prompt)
            item["prompt_mask"].append(np.ones(len(prompt), dtype=np.int32))

            # Pad the completion part as needed
            completion_pad_len = max(0, token_setup_len - prompt_length - len(completion))
            padded_completion = np.concatenate([
                np.array(completion),
                np.ones(completion_pad_len, dtype=np.int32) * pad_token_id,
            ])
            padded_completion_mask = np.concatenate([
                np.ones(len(completion), dtype=np.int32),
                np.zeros(completion_pad_len, dtype=np.int32)
            ])

            item["completion_ids"].append(padded_completion)
            item["completion_mask"].append(padded_completion_mask)

        for i in range(len(item["tokens"])):
            label_item = np.concatenate(
                [
                    np.array(item["prompt_ids"][i]),
                    np.full(
                        max(0, len(item["prompt_ids"][i])),
                        -100,
                        dtype=np.int32,
                    ),
                ]
            )
            item["tokens"][i] = np.concatenate(
                [
                    np.array(item["tokens"][i]),
                    np.zeros(
                        max(0, token_setup_len - len(item["tokens"][i])), dtype=np.int32
                    ),
                ]
            )
            prompt_ids.append(item["prompt_ids"][i])
            prompt_mask.append(item["prompt_mask"][i])
            completion_ids.append(item["completion_ids"][i])
            completion_mask.append(item["completion_mask"][i])
            advantages.append(item["scores"][i])
    # combine all lists into tensors
    prompt_ids_batches = []
    prompt_mask_batches = []
    completion_ids_batches = []
    completion_mask_batches = []
    advantages_batches = []

    for i in range(len(prompt_ids) // batch_size):
        prompt_ids_batches.append(
            torch.tensor(
                np.stack(prompt_ids[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        prompt_mask_batches.append(
            torch.tensor(
                np.stack(prompt_mask[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        completion_ids_batches.append(
            torch.tensor(
                np.stack(completion_ids[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        completion_mask_batches.append(
            torch.tensor(
                np.stack(completion_mask[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        advantages_batches.append(
            torch.tensor(
                np.stack(advantages[i * batch_size : (i + 1) * batch_size], axis=0)
            ).view(-1, 1)
        )
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
