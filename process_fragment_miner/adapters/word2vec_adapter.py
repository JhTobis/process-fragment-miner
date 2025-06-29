from gensim.models import Word2Vec
from typing import List


class Word2VecAdapter:
    def __init__(self, traces: List[List[str]], vector_size=64, window=24, min_count=1, workers=4):
        """
        Trains a Word2Vec model on the provided traces.

        Args:
            traces (List[List[str]]): Training traces.
        """
        self.model = Word2Vec(
            sentences=traces,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1  # Skip-gram model
        )

    def similarity(self, word1: str, word2: str) -> float:
        """
        Returns the cosine similarity between two activities.

        Args:
            word1 (str), word2 (str): Activity labels

        Returns:
            float: Cosine similarity or raises KeyError if OOV
        """
        return self.model.wv.similarity(word1, word2)

    def contains(self, word: str) -> bool:
        """
        Checks if a word is in the vocabulary.

        Args:
            word (str): Activity label

        Returns:
            bool
        """
        return word in self.model.wv