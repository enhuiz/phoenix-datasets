class LookupTable:
    def __init__(self, words=None, symbols=None, allow_unk=False):
        """
        Args:
            words: all the words, unsorted, allows duplications
            symbols: no duplications
        """
        assert words is None or symbols is None, "Specify either words or symbols."

        self.allow_unk = allow_unk

        if symbols is None:
            symbols = sorted(set(words))

        assert len(symbols) == len(set(symbols)), "Symbols contain duplications."

        self.symbols = symbols
        self.mapping = {symbol: i for i, symbol in enumerate(symbols)}

    def __call__(self, symbol):
        if symbol in self.mapping:
            return self.mapping[symbol]
        elif self.allow_unk:
            return len(self) - 1
        raise KeyError(symbol)

    def __getitem__(self, i):
        if i < len(self.symbols):
            return self.symbols[i]
        elif i == len(self.symbols):
            return "unk"
        else:
            raise IndexError(f"Index {i} out of range.")

    def __len__(self):
        return len(self.mapping) + int(self.allow_unk)

    def __str__(self):
        unk = {"unk": len(self) - 1} if self.allow_unk else {}
        return str({**self.mapping, **unk})
