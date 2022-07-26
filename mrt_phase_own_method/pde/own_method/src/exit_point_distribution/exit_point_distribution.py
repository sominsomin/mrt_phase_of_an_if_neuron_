import pickle


class ExitPointDistribution:
    def __init__(self, v, a, p):
        self.v = v
        self.a = a
        self.p = p

    def __len__(self):
        return len(self.p)

    def __getitem__(self, index):
        return self.p[index]

    def __setitem__(self, index, value):
        self.p[index] = value

    def __repr__(self):
        return f'{self.p}'

    def save(self, filename):
        """
        save object to file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        """
        load from pickle
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)