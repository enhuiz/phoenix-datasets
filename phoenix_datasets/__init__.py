from .datasets import VideoTextDataset
from .corpora import PhoenixCorpus, PhoenixTCorpus


class PhoenixVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixCorpus


class PhoenixTVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixTCorpus
