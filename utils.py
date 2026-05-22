import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_layer_list(spec):
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def decode_sequences(tokenizer, sequences):
    if sequences is None:
        return []
    pad_id = tokenizer.pad_token_id
    sequences = np.where(sequences == -100, pad_id, sequences)
    return tokenizer.batch_decode(sequences, skip_special_tokens=True)


def remove_best_checkpoint(output_dir, filename="best_model.pt"):
    path = os.path.join(output_dir, filename)
    if os.path.isfile(path):
        os.remove(path)
