import os
import subprocess
import csv
import tempfile
import pandas as pd
import contextlib
import shutil
from functools import partial
from pathlib import Path

from .corpora import Corpus, PhoenixCorpus


@contextlib.contextmanager
def working_directory(new_dir):
    old_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(old_dir)


class PhoenixEvaluator:
    def __init__(self, root):
        root = Path(root)
        sclite_installer = (
            root / "evaluation" / "NIST-sclite_sctk-2.4.0-20091110-0958.tar.bz2"
        )
        if shutil.which("sclite") is None:
            raise RuntimeError(
                '"sclite" not found. Please install it and then add it to PATH first. '
                "It is required officially for WER calculation. "
                f'Check "{sclite_installer}" for detail.'
            )
        self.corpus = PhoenixCorpus(root)
        self.folder = (root / "evaluation").absolute()

    @staticmethod
    def make_ctm(ids, sentences):
        ctm = []
        for id_, sentence in zip(ids, sentences):
            words = sentence.split()
            if not words:
                words = ["[EMPTY]"]
            for j, word in enumerate(words):
                # due to the limitation of the phoenix's script,
                # j must be string sortable.
                ctm.append([id_, 1, f"{j:06d}", 1, word])
        ctm = pd.DataFrame(ctm)
        return ctm

    def fix_scripts(self):
        """
        On ubuntu 20.04, this scripts has some errors as commented below.
        Call this in the tmpdir to fix them without modify the original script.
        """
        self.fix_main_script()
        self.fix_mergectmstm()

    def fix_main_script(self):
        """
        Change line 23 from mergectmstm.py => ./mergectmstm.py
        Change line 28, add dtl report
        """
        path = "./evaluatePhoenix2014.sh"
        with open(path, "r") as f:
            content = f.read()
        content = content.replace("mergectmstm.py", "./mergectmstm.py")
        content = content.replace("-o sgml sum rsum pra", "-o sgml sum rsum pra dtl")
        with open(path, "w") as f:
            f.write(content)

    def fix_mergectmstm(self):
        """Fix python indent."""
        path = "./mergectmstm.py"
        try:
            result = subprocess.check_output([path], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            output = e.output.decode("utf8")
            if "TabError" in output:
                with open(path, "r") as f:
                    content = f.read()
                with open(path, "w") as f:
                    f.write(content.replace("\t", " " * 8))
            elif "out of range" not in output:
                print(e.output)
                raise e

    @staticmethod
    def make_trn(ids, sentences, speakers):
        trn = []
        for id_, sentence, speaker in zip(ids, sentences, speakers):
            trn.append([sentence, f"({speaker}-{id_})"])
        trn = pd.DataFrame(trn)
        return trn

    @staticmethod
    def parse_dtl(dtl):
        dtl = dtl.split("WORD RECOGNITION PERFORMANCE")[1]
        dtl = dtl.strip()
        dtl = dtl.splitlines()[:7]
        dtl = filter(lambda l: l, dtl)

        def parse_line(l):
            name, tail = l.split("=")
            value, _ = tail.split("%")
            name = name.lower().replace("percent", "").strip()
            value = float(value.strip())
            return name, value

        dtl = [parse_line(l) for l in dtl]
        dtl = {n: v for n, v in dtl if n != "correct"}

        return dtl

    def link_files(self):
        for name in [
            "evaluatePhoenix2014.sh",
            "mergectmstm.py",
            "phoenix2014-groundtruth-dev.stm",
            "phoenix2014-groundtruth-test.stm",
        ]:
            shutil.copy(self.folder / name, name)

    def evaluate(self, split, hyps, reports=["sum", "dtl", "pra"]):
        df = self.corpus.load_data_frame(split)
        df = df.sort_values("id")

        if len(df) != len(hyps):
            print(f"Warning: #hyps should be {len(df)} for {split}, got {len(hyps)}.")

        out = {}
        ctm = self.make_ctm(df["id"], hyps)
        shell = partial(subprocess.check_output, shell=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            with working_directory(tmpdir):
                self.link_files()
                self.fix_scripts()
                ctm.to_csv("hypothesis.ctm", index=None, header=None, sep=" ")
                shell([f"./evaluatePhoenix2014.sh hypothesis.ctm {split}"])
                for r in reports:
                    ext = "sys" if r == "sum" else r
                    out[r] = shell([f"cat out.hypothesis.ctm.{ext}"]).decode("utf8")

        if "dtl" in reports:
            out["parsed_dtl"] = self.parse_dtl(out["dtl"])

        return out


if __name__ == "__main__":
    evaluator = PhoenixEvaluator("data/phoenix-2014-multisigner")
    hyp = evaluator.corpus.load_data_frame("dev")["annotation"].apply(" ".join).tolist()
    hyp[0] = "THIS SENTENCE IS WRONG"
    results = evaluator.evaluate("dev", hyp)
    print(results["parsed_dtl"])
    print(results["sum"])
