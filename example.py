from phoenix_datasets import PhoenixVideoTextDataset

from torch.utils.data import DataLoader

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
    label = batch["label"]

    # Do per-frame augmentation (e.g. normalization, cropping) here if needed.
    # kornia will be a good tool for this
    # video = augment(video)

    assert len(video) == len(label)
    print(len(video))
    print(video[0].shape)
    print(label[0].shape)

    break
