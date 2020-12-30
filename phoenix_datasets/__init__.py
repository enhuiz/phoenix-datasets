from .datasets import VideoTextDataset
from .corpora import PhoenixCorpus, PhoenixTCorpus
from .evaluator import PhoenixEvaluator


class PhoenixVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixCorpus


class PhoenixTVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixTCorpus
