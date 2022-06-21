import pickle
import os


class T_mean:
    T = None
    v = None
    a = None

    def __init__(self):
        pass

    def save(self):
        """
        save object to file
        """
        filename = f''

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        print(f'file saved to {filename}')

    @classmethod
    def load(cls, filename: str):
        """
        load isochrone from object
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)