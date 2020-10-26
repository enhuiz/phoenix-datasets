from .datasets import VideoTextDataset
from .corpus import PhoenixCorpus


class PhoenixVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixCorpus
