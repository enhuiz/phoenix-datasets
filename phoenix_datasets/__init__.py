from .datasets import VideoTextDataset
from .corpora import PhoenixCorpus, PhoenixTCorpus
from .evaluators import PhoenixEvaluator


class PhoenixVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixCorpus


class PhoenixTVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixTCorpus
