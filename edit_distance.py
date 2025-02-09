class EditDistance:
    def __init__(self):
        self.store = {}

    def _distance(self, s, t):
        if len(s) == 0:
            d = len(t)
        elif len(t) == 0:
            d = len(s)
        else:
            a, b, c = tuple(self.store[pair] if pair in self.store else self._distance(*pair)
                            for pair in ((s[:-1], t), (s, t[:-1]), (s[:-1], t[:-1])))
            d = min([a + 1, b + 1, c + (0 if s[-1] == t[-1] else 1)])
        if (s, t) not in self.store:
            self.store[s, t] = d
        return d

    def distance(self, s, t):
        self.store = {}
        return self._distance(s, t)
