# PHOENIX Datasets 🐦

## Introduction

[PHOENIX-2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) and [PHOENIX-2014-T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) are popular large scale German sign language datasets developed by Human Language Technology & Pattern Recognition Group from RWTH Aachen University, Germany. This package provides a PyTorch dataset wrapper for those two datasets to make the building of PyTorch model on these two datasets easier.

## Installation

```bash
pip install git+https://github.com/enhuiz/phoenix-datasets
```

## Example Usage

```python
from phoenix_datasets import PhoenixVideoTextDataset

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

dtrain = PhoenixVideoTextDataset(
    # your path to this folder, download it from official website first.
    root="data/phoenix-2014-multisigner",
    split="train",
    p_drop=0.5,
    random_drop=True,
)

vocab = dtrain.vocab

print("Vocab", vocab)

dl = DataLoader(dtrain, collate_fn=dtrain.collate_fn)

for batch in dl:
    video = batch["video"]
    text = batch["text"]

    # Do per-frame augmentation (e.g. normalization, cropping) here if needed.
    # kornia will be a good tool for this
    # video = augment(video)

    assert len(video) == len(text)
    print(len(video))
    print(video[0].shape)
    print(text[0].shape)

    break
```

## Supported Features

- [x] Load automatic alignment for PHOENIX-2014
- [x] Random/evenly frame dropping augmentation

## TODOs

- [ ] Implement Corpus for PHOENIX-2014-T
