from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from process_fragment_miner.adapters.word2vec_adapter import Word2VecAdapter

class BaseScorer:
    def score(self, trace: List[str]) -> float:
        raise NotImplementedError("Must implement score(trace)")


class BigramScorer(BaseScorer):
    """
    Scores traces based on bigram likelihoods derived from training traces.
    """

    def __init__(self, traces: List[List[str]], smoothing: float = 1.0):
        """
        Args:
            traces (List[List[str]]): Training traces
            smoothing (float): Laplace smoothing factor
        """
        self.smoothing = smoothing
        self.unigrams, self.bigrams, self.total_unigrams = self._build_ngram_counts(traces)

    def score(self, trace: List[str]) -> float:
        """
        Scores a single trace using smoothed bigram likelihood.
        """
        score = self._compute_trace_likelihood(trace)
        return score if score is not None else float('-inf')

    def _build_ngram_counts(self, traces: List[List[str]]):
        unigram_counts = defaultdict(int)
        bigram_counts = defaultdict(int)
        total_unigrams = 0

        for trace in traces:
            for i, word in enumerate(trace):
                unigram_counts[word] += 1
                total_unigrams += 1
                if i > 0:
                    bigram_counts[(trace[i - 1], word)] += 1

        return unigram_counts, bigram_counts, total_unigrams

    def _compute_trace_likelihood(self, trace: List[str]) -> float:
        vocab_size = len(self.unigrams)
        likelihood = 1.0

        for i, word in enumerate(trace):
            if i == 0:
                # Probability of first word (unigram)
                p = (self.unigrams[word] + self.smoothing) / (self.total_unigrams + self.smoothing * vocab_size)
            else:
                prev_word = trace[i - 1]
                p = (self.bigrams[(prev_word, word)] + self.smoothing) / \
                    (self.unigrams[prev_word] + self.smoothing * vocab_size)
            likelihood *= p

        return likelihood

class DependencyScorer(BaseScorer):
    """
    Scores traces based on dependency matrix values between activities.
    """
    def __init__(self, dependency_matrix: Dict[str, Dict[str, float]]):
        self.matrix = dependency_matrix

    def score(self, trace: List[str]) -> float:
        if len(trace) < 2:
            return float('-inf')
        score = 1.0
        for i in range(len(trace) - 1):
            a, b = trace[i], trace[i + 1]
            dep = self.matrix.get(a, {}).get(b, None)
            if dep is None:
                return float('-inf')
            score *= dep
        return score

class SimilarityScorer(BaseScorer):
    """
    Scores traces based on average pairwise Word2Vec similarity between activities.
    """

    def __init__(self, traces: List[List[str]], remove_loops: bool = False):
        """
        Initializes the scorer using a Word2VecAdapter trained on traces.

        Args:
            train_traces: List of training traces
            remove_loops: Whether to exclude traces with loops in bulk scoring
        """
        self.model = Word2VecAdapter(traces)
        self.remove_loops = remove_loops

    def score(self, trace: List[str]) -> float:
        """
        Computes the average similarity of consecutive pairs in a trace.

        Args:
            trace: A list of activity labels

        Returns:
            float similarity score or -inf if any activity is OOV
        """
        if len(trace) < 2:
            return 0.0

        try:
            similarities = [
                self.model.similarity(trace[i], trace[i + 1])
                for i in range(len(trace) - 1)
            ]
            return float(np.mean(similarities))
        except KeyError:
            return float('-inf')

    def score_traces(self, traces: List[List[str]]) -> List[Tuple[List[str], float]]:
        """
        Scores and ranks a list of traces.

        Returns:
            List of (trace, score) sorted descending
        """
        def has_loops(trace):
            return len(set(trace)) < len(trace)

        if self.remove_loops:
            traces = [t for t in traces if not has_loops(t)]

        seen = set()
        unique_traces = []
        for t in traces:
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                unique_traces.append(t)

        scored = [(t, self.score(t)) for t in unique_traces]
        return sorted(scored, key=lambda x: x[1], reverse=True)