from .datasets import VideoTextDataset
from .corpus import PhoenixCorpus, PhoenixTCorpus


class PhoenixVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixCorpus


class PhoenixTVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixTCorpus
