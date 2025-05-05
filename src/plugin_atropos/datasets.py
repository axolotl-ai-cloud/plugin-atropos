import json
import math
import queue
import random
import threading
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from datasets import IterableDataset
from google.auth.transport import requests
from tenacity import retry, stop_after_attempt, wait_exponential

class RemoteIterableDataset(IterableDataset):
    def __init__(self, *args, ttl=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.ttl = ttl  # Time-to-live for blocking on the queue

    def add_data(self, data):
        """
        Method to add data to the queue.
        """
        self.data_queue.put(data)

    def stop(self):
        """
        Method to signal that no more data will be added and iteration should stop
        when the queue is empty.
        """
        self.stop_event.set()

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_event.is_set() and self.data_queue.empty():
            # If stop has been signaled and the queue is empty, end the iteration
            raise StopIteration

        try:
            # Attempt to get data from the queue with blocking up to the TTL
            data = self.data_queue.get(timeout=self.ttl)
        except queue.Empty:
            # If no data is available within the TTL, raise StopIteration
            raise StopIteration

        return data

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_batch():
    data = requests.get("http://localhost:8000/batch", timeout=10).json()
    return data

def pad_data_to_good_offset(data, batch_size: int):
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
    input_ids = list()
    labels = list()
    advantages = list()
    lengths = list()
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
        for i in range(len(item["tokens"])):
            lengths.append(
                math.ceil((len(item["tokens"][i]) - 1) / good_multiple)
                * good_multiple
            )
            label_item = np.concatenate(
                [
                    np.array(item["masks"][i]),
                    np.full(
                        max(0, token_setup_len - len(item["tokens"][i])),
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
            input_ids.append(item["tokens"][i][:-1])
            labels.append(label_item[1:])
            advantages.append(item["scores"][i])
    # combine all lists into tensors
    token_batches = []
    label_batches = []
    advantage_batches = []
    for i in range(len(input_ids) // batch_size):
        token_batches.append(
            torch.tensor(
                np.stack(input_ids[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        label_batches.append(
            torch.tensor(
                np.stack(labels[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        advantage_batches.append(
            torch.tensor(
                np.stack(advantages[i * batch_size : (i + 1) * batch_size], axis=0)
            ).view(-1, 1)
        )
    return token_batches, label_batches, advantage_batches

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
            batches.append(pad_data_to_good_offset(data, batch_size))
            # return batches
        elif len(batches) > 0:
            # Return the batches
            return batches
        else:
            time.sleep(1)

def data_producer():
    seen_keys = defaultdict(set)
    while True:
        ds = get_batch()
        for sample in ds:
            yield sample['id'], sample['data']
