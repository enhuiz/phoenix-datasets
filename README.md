# PHOENIX Datasets 🐦

## Introduction

[PHOENIX-2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) and [PHOENIX-2014-T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) are popular large scale German sign language datasets developed by Human Language Technology & Pattern Recognition Group from RWTH Aachen University, Germany. This package provides a PyTorch dataset wrapper for those two datasets to make the building of PyTorch model on these two datasets easier.

## Installation

```bash
pip install git+https://github.com/enhuiz/phoenix-datasets
```

## Example Usage

### Dataset

```python
from phoenix_datasets import PhoenixVideoTextDataset

from torch.utils.data import DataLoader

dtrain = PhoenixVideoTextDataset(
    # your path to this folder, download it from official website first.
    root="data/phoenix-2014-multisigner",
    split="train",
    p_drop=0.5,
    random_drop=True,
    random_crop=True,
    base_size=[256, 256]
    crop_size=[224, 224],
)

vocab = dtrain.vocab

print("Vocab", vocab)

dl = DataLoader(dtrain, collate_fn=dtrain.collate_fn)

for batch in dl:
    video = batch["video"]
    label = batch["label"]
    signer = batch["signer"]

    assert len(video) == len(label)

    print(len(video))
    print(video[0].shape)
    print(label[0].shape)
    print(signer)

    break
```

### Evaluation

Go to `phoenix-2014-multisigner/evaluation/NIST-sclite_sctk-2.4.0-20091110-0958.tar.bz2` to install `sclite` (the official tool for WER calculation) first and then put it in your PATH.

```python
from phoenix_datasets.evaluators import PhoenixEvaluator

evaluator = PhoenixEvaluator("data/phoenix-2014-multisigner")
hyp = evaluator.corpus.load_data_frame("dev")["annotation"].apply(" ".join).tolist()
hyp[0] = "THIS SENTENCE IS WRONG"
results = evaluator.evaluate("dev", hyp)
print(results["parsed_dtl"])
print(results["sum"])
```

## Supported Features

- [x] Load the automatic alignments for PHOENIX-2014
- [x] Randomly/evenly frame dropping augmentation
- [x] Evaluation for Phoenix-2014
- [x] Language Model

## TODOs

- [ ] Implement Corpus and evaluation for PHOENIX-2014-T
