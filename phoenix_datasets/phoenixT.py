def load_corpus(root, split):
    # may be 2014-T
    filename = f"PHOENIX-2014-T.{split}.corpus.csv"
    filename = filename.replace("train", "train-complex-annotation")
    df = pd.read_csv(Path(dirname, filename), sep="|")
    df = df.rename(columns={"speaker": "signer", "name": "id", "orth": "reference"})
