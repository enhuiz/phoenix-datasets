import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import partial
from PIL import Image
from torchvision import transforms
from collections import defaultdict


def load_pil(path):
    # convert back to numpy as tensor in dataloader may cause
    # fd problem: https://github.com/pytorch/pytorch/issues/11201
    # and you probably don't want:
    # torch.multiprocessing.set_sharing_strategy('file_system')
    return transforms.functional.to_tensor(Image.open(path)).numpy()


class VideoTextDataset(Dataset):
    Corpus = None

    def __init__(self, root, split, p_drop=0, random_drop=True, vocab=None):
        """
        Args:
            root: Root to the data set, e.g. the folder contains features/ annotations/ etc..
            split: data split, e.g. train/dev/test
            p_drop: proportion of frame dropping.
            random_drop: if True, random drop else evenly drop.
            vocab: gloss to index (categorize).
        """
        assert 0 <= p_drop <= 1, f"p_drop value {p_drop} is out of range."
        assert (
            self.Corpus is not None
        ), f"Corpus is not defined in the derived class {self.__class__.__name__}."

        self.corpus = self.Corpus(root)
        self.random_drop = random_drop
        self.p_drop = p_drop

        self.data_frame = self.corpus.load_data_frame(split)
        self.vocab = vocab or self.corpus.create_vocab()

    def sample_indices(self, n):
        p_kept = 1 - self.p_drop
        if self.random_drop:
            indices = np.arange(n)
            np.random.shuffle(indices)
            indices = indices[: int(n * p_kept)]
            indices = sorted(indices)
        else:
            indices = np.arange(0, n, 1 / p_kept)
            indices = np.round(indices)
            indices = np.clip(indices, 0, n - 1)
            indices = indices.astype(int)
        return indices

    @staticmethod
    def select_elements(l, indices):
        return [l[i] for i in indices]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        sample = {**self.data_frame.iloc[index].to_dict()}  # copy
        frames = self.corpus.get_frames(sample, "fullFrame-210x260px")

        indices = self.sample_indices(len(frames))

        frames = self.select_elements(frames, indices)
        frames = np.stack(list(map(load_pil, frames)))

        texts = list(map(self.vocab, sample["annotation"]))

        return {
            "video": frames,
            "text": texts,
        }

    @staticmethod
    def collate_fn(batch):
        collated = defaultdict(list)
        for sample in batch:
            collated["video"].append(torch.tensor(sample["video"]).float())
            collated["text"].append(torch.tensor(sample["text"]).long())
        return dict(collated)
