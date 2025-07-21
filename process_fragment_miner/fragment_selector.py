import math
import heapq
import os
from collections import defaultdict
import psutil


def _memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # in MB


def _convert_to_bitmasked(subtraces):
    all_events = set(e for _, trace in subtraces for e in trace)
    event_to_index = {e: i for i, e in enumerate(all_events)}

    bitmasked = []
    for score, trace in subtraces:
        mask = 0
        for e in trace:
            mask |= 1 << event_to_index[e]
        bitmasked.append((score, trace, mask))
    return bitmasked


def _score_trace_set(scores, bitmask, score_agg, alpha):
    if not scores:
        return float('-inf')

    if score_agg == "sum":
        val = sum(scores)
    elif score_agg == "mean":
        val = sum(scores) / len(scores)
    elif score_agg == "log_likelihood":
        if any(s <= 0 for s in scores):
            return float('-inf')
        val = sum(math.log(s) for s in scores)
    else:
        raise ValueError(f"Unsupported score_agg: {score_agg}")

    if alpha is not None:
        coverage = bin(bitmask).count("1")
        val += alpha * coverage

    return val


def _generate_candidate_state(score, trace, trace_mask, used_mask, trace_list, score_list, score_agg, alpha):
    if used_mask & trace_mask != 0:
        return None

    new_mask = used_mask | trace_mask
    new_traces = trace_list + [trace]
    new_scores = score_list + [score]
    new_val = _score_trace_set(new_scores, new_mask, score_agg, alpha)

    return new_mask, new_val, new_traces, new_scores


def _dp_solver(subtraces, score_agg, alpha, return_details, max_memory_mb):
    dp = defaultdict(lambda: (float('-inf'), [], []))
    dp[0] = (0, [], [])

    for score, trace, trace_mask in subtraces:
        updates = []
        for used_mask, (curr_val, trace_list, score_list) in dp.items():
            result = _generate_candidate_state(
                score, trace, trace_mask, used_mask,
                trace_list, score_list, score_agg, alpha
            )
            if result:
                new_mask, new_val, new_traces, new_scores = result
                updates.append((new_mask, new_val, new_traces, new_scores))

        for new_mask, new_val, new_traces, new_scores in updates:
            if new_val > dp[new_mask][0]:
                dp[new_mask] = (new_val, new_traces, new_scores)
        
        if _memory_usage_mb() > max_memory_mb:
            return None  # fallback to beam

    best_val, best_traces, best_scores = max(dp.values(), key=lambda x: x[0])
    return (best_val, best_traces, best_scores) if return_details else (best_val, best_traces)


def _beam_solver(subtraces, score_agg, alpha, beam_size, return_details):
    beam = [(0, 0, [], [], 0)]  # (neg_val, used_mask, trace_list, score_list, coverage)

    for score, trace, trace_mask in subtraces:
        seen = {}
        for neg_val, used_mask, trace_list, score_list, _ in beam:
            result = _generate_candidate_state(
                score, trace, trace_mask, used_mask,
                trace_list, score_list, score_agg, alpha
            )
            if result:
                new_mask, new_val, new_traces, new_scores = result
                tup = (-new_val, new_mask, new_traces, new_scores, bin(new_mask).count("1"))
                if new_mask not in seen or -new_val > seen[new_mask][0]:
                    seen[new_mask] = tup

        combined = list(beam) + list(seen.values())
        beam = heapq.nsmallest(beam_size, combined)

    best = min(beam)
    return (-best[0], best[2], best[3]) if return_details else (-best[0], best[2])


def get_best_disjoint_subset(
    subtraces,
    score_agg="sum",
    alpha=None,
    beam_size=100,
    max_memory_mb=500,
    return_details=False,
    method="auto"
):
    """
    Selects the best disjoint subset of traces using DP or beam search.

    Args:
        subtraces (list): List of (score, trace) tuples.
        score_agg (str): One of 'sum', 'mean', 'log_likelihood'.
        alpha (float): Optional bonus for number of unique elements.
        beam_size (int): Beam width for beam search.
        max_memory_mb (int): Max memory allowed for DP (fallbacks to beam).
        return_details (bool): Whether to return individual trace scores.
        method (str): "dp", "beam", or "auto".

    Returns:
        tuple: (score, [subtraces]) or (score, [subtraces], [scores])
    """
    bitmasked = _convert_to_bitmasked(subtraces)

    if method == "dp" or method == "auto":
        result = _dp_solver(bitmasked, score_agg, alpha, return_details, max_memory_mb)
        if result is not None:
            return result + (("dp",) if return_details else ())
        elif method == "dp":
            raise RuntimeError("DP solver failed due to memory constraints.")
        
        print("⚠️  Memory exceeded. Switching to beam search.")

    if method == "beam" or method == "auto":
        return _beam_solver(bitmasked, score_agg, alpha, beam_size, return_details) + (("beam",) if return_details else ())

    raise ValueError(f"Invalid method: {method}. Choose from 'dp', 'beam', or 'auto'.")