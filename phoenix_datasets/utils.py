class LookupTable:
    def __init__(self, keys, allow_unk=False):
        """
        Args:
            keys: the tokens.
            allow_unk: if True, #tokens will increased by one as unk is appended. OOV will not raise an error.
        """
        self.allow_unk = allow_unk
        keys = sorted(set(keys))
        self.table = {key: i for i, key in enumerate(keys)}

    def __call__(self, key):
        if key in self.table:
            return self.table[key]
        elif self.allow_unk:
            return len(self) - 1
        raise KeyError(key)

    def __len__(self):
        return len(self.table) + int(self.allow_unk)

    def __str__(self):
        unk = {"unk": len(self) - 1} if self.allow_unk else {}
        return str({**self.table, **unk})
