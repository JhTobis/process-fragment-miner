from typing import List
from process_fragment_miner.adapters.pm4py_adapter import extract_dependency_graph
from process_fragment_miner.subtrace_extractor import extract_top_scoring_subtraces
from process_fragment_miner.fragment_selector import get_best_disjoint_subset
from process_fragment_miner.scorer import BaseScorer  # Optional: type hint
from process_fragment_miner.scorer_factory import ScorerFactory


class ProcessFragmentMiner:
    def __init__(self, event_log, scorer="bigram", scorer_kwargs=None, dependency_threshold=0.2):
        """
        Initializes the miner.

        Args:
            event_log: PM4Py event log object
            scorer: str (e.g. 'bigram', 'dependency', 'similarity'),
                    or an instance of BaseScorer
            dependency_threshold: Threshold for edge inclusion in dependency graph
        """
        self.event_log = event_log
        self.dependency_threshold = dependency_threshold
        self.dependencies = extract_dependency_graph(event_log, threshold=dependency_threshold)

        self._traces = None
        self._event_names = None

        # Create or assign scorer
        if isinstance(scorer, str):
            self._scorer = ScorerFactory.create(scorer, miner=self, **(scorer_kwargs or {}))
        elif callable(scorer):
            self._scorer = scorer(self)
        else:
            self._scorer = scorer

    @property
    def scorer(self):
        return self._scorer

    def get_traces(self):
        """
        Converts PM4Py event log to list of activity traces (List[List[str]])
        """
        if self._traces is None:
            self._traces = [
                [event["concept:name"] for event in trace]
                for trace in self.event_log
            ]
        return self._traces

    def get_event_names(self):
        if self._event_names is None:
            self._event_names = {event["concept:name"] for trace in self.event_log for event in trace}
        return self._event_names

    def extract_subtraces(self, max_depth=5, min_depth=2, top_k=10):
        """
        Extracts top-scoring subtraces from the dependency graph using DFS.

        Args:
            max_depth (int): Maximum depth of each subtrace.
            min_depth (int): Minimum length of valid subtraces.
            top_k (int): Maximum subtraces to keep per start node.

        Returns:
            List of (score, trace) tuples.
        """
        return extract_top_scoring_subtraces(
            dependencies=self.dependencies,
            score_function=self.scorer.score,
            threshold=self.dependency_threshold,
            max_depth=max_depth,
            min_depth=min_depth,
            top_k=top_k
        )

    def mine_best_fragments(
        self,
        subtraces,
        score_agg="sum",
        alpha=None,
        beam_size=100,
        max_memory_mb=500,
        return_details=False,
        method="auto",
        ensure_coverage=True
    ):
        """
        Finds the best disjoint subset of subtraces using DP or beam search.

        Args:
            subtraces (list): List of (score, trace) tuples.
            score_agg (str): Aggregation method: 'sum', 'mean', 'log_likelihood'.
            alpha (float): Optional reward per unique event covered.
            beam_size (int): Beam width (used in beam search).
            max_memory_mb (int): Max memory usage for DP before fallback.
            return_details (bool): Whether to include individual trace scores.
            method (str): "dp", "beam", or "auto" (default: auto).
            ensure_coverage (bool): If True, guarantees every event name appears in at least one fragment.

        Returns:
            tuple: (total_score, [traces], [individual_scores]?)
        """
        result = get_best_disjoint_subset(
            subtraces=subtraces,
            score_agg=score_agg,
            alpha=alpha,
            beam_size=beam_size,
            max_memory_mb=max_memory_mb,
            return_details=return_details,
            method=method
        )

        if return_details:
            score, best_traces, trace_scores = result
        else:
            score, best_traces = result
            trace_scores = None

        if ensure_coverage:
            fragments = self._ensure_full_coverage(best_traces)
        else:
            fragments = best_traces

        if return_details:
            return score, fragments, trace_scores
        else:
            return score, fragments
        
    def _ensure_full_coverage(self, fragments: List[List[str]]) -> List[List[str]]:
        """
        Ensures all event names in the log are represented in the returned fragments.
        """
        covered = {act for fragment in fragments for act in fragment}
        all_events = self.get_event_names()
        missing = all_events.difference(covered)

        if not missing:
            return fragments

        return fragments + [list(missing)]