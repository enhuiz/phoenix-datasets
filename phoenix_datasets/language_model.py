import pkg_resources
import re
import arpa
import xmltodict
from itertools import product
from collections import defaultdict


def unk_set():
    return {"[UNKNOWN]"}


def create_gloss_mapping():
    """
    This maps modified glosses to original glosses.
    """
    with pkg_resources.resource_stream(__name__, f"data/3state.lex") as f:
        d = xmltodict.parse(f.read())

    mapping = defaultdict(set)

    for kv in d["lexicon"]["lemma"][4:]:
        if "orth" not in kv or "phon" not in kv:
            continue

        orths = kv["orth"]
        phons = kv["phon"]

        if not isinstance(kv["orth"], list):
            orths = [orths]

        if not isinstance(kv["phon"], list):
            phons = [phons]

        for orth in orths:
            for phon in phons:
                phon = phon.split()[0]
                if phon[-1].isdigit():
                    phon = phon[:-1]
                mapping[phon].add(orth)

    return defaultdict(unk_set, mapping)


class SRILM:
    def __init__(self, path, vocab):
        self.lm = arpa.loadf(path)[0]
        self.vocab = vocab
        self.mapping = create_gloss_mapping()

    def __call__(self, indices):
        """
        probability p(end|in, the) = lm("in the end")
        """
        sentences = product(*[self.mapping[self.vocab[i]] for i in indices])
        return max(map(self.p, sentences))

    def p(self, sentence):
        try:
            return self.lm.p(("<s>",) + sentence)
        except Exception as e:
            return 0
