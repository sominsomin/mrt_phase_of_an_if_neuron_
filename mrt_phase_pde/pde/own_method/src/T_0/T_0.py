import pickle


class T_0:
    def __init__(self, v, a, T):
        self.v = v
        self.a = a

        self.T = T

    def __getitem__(self, index):
        return self.T[index]

    def __repr__(self):
        return f'{self.T}'

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