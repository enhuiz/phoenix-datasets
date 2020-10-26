import pandas as pd
from pathlib import Path
from pandarallel import pandarallel
from functools import partial

from .utils import LookupTable

pandarallel.initialize(verbose=1)


class Corpus:
    def __init__(self, root):
        pass

    def load_data_frame(self, split):
        raise NotImplementedError

    def create_vocab(self, data_frame=None):
        df = data_frame or self.load_data_frame("train")
        sentences = df["annotation"].to_list()
        return LookupTable([gloss for sentence in sentences for gloss in sentence])


class PhoenixCorpus(Corpus):
    def __init__(self, root):
        self.root = Path(root)

    def load_alignment(self):
        dirname = self.root / "annotations" / "automatic"

        # important to literally read NULL instead read it as nan
        read = partial(pd.read_csv, sep=" ", na_filter=False)
        ali = read(dirname / "train.alignment", header=None, names=["id", "classlabel"])
        cls = read(dirname / "trainingClasses.txt")

        df = pd.merge(ali, cls, how="left", on="classlabel")
        del df["classlabel"]

        df["gloss"] = df["signstate"].apply(lambda s: s.rstrip("012"))

        df["id"] = df["id"].parallel_apply(lambda s: "/".join(s.split("/")[3:-2]))
        grouped = df.groupby("id")

        gdf = grouped["gloss"].agg(" ".join)
        sdf = grouped["signstate"].agg(" ".join)

        df = pd.merge(gdf, sdf, "inner", "id")

        assert (
            len(df) == 5671
        ), f"Alignment file is not correct, expect to have 5671 entries but got {len(df)}."

        return df

    def load_data_frame(self, split, aligned_annotation=False):
        """Load corpus."""
        path = self.root / "annotations" / "manual" / f"{split}.corpus.csv"
        df = pd.read_csv(path, sep="|")
        df["annotation"] = df["annotation"].apply(str.split)

        if split == "train" and aligned_annotation:
            # append alignment to data frame
            # note that only train split has alignment
            adf = self.load_alignment()
            adf = adf.rename({"gloss": "annotation"}, axis=1)
            adf = adf["annotation"]
            del df["annotation"]
            df = pd.merge(df, adf, "left", "id")

        df["folder"] = split + "/" + df["folder"].apply(lambda s: s.rsplit("/", 1)[0])

        return df

    def get_frames(self, sample, type):
        frames = (self.root / "features" / type / sample["folder"]).glob("*.png")
        return sorted(frames)


class PhoenixTCorpus(PhoenixCorpus):
    def __init__(self):
        # TODO:
        raise NotImplementedError
