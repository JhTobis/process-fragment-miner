import heapq
import math

def extract_top_scoring_subtraces(
    dependencies, score_function, threshold, max_depth, min_depth, top_k
):
    """
    Extract high-scoring subtraces from a dependency graph.

    Args:
        dependencies (dict): {node: {next_node: strength, ...}}
        score_function (function): Function to score a trace (list of nodes).
        threshold (float): Minimum edge strength to follow.
        max_depth (int): Max depth of traces to explore.
        min_depth (int): Minimum depth to consider a trace valid.
        top_k (int): Keep only top_k highest scoring subtraces per start node.

    Returns:
        list of tuples: [(score, trace), ...] where trace is a list of nodes.
    """
    all_subtraces = []

    for start_node in dependencies:
        stack = [(start_node, [start_node], 1)]  # (current_node, trace_so_far, depth)
        visited = set()
        top_traces_heap = []  # Min-heap of (score, trace)

        while stack:
            current_node, trace, depth = stack.pop()

            # Avoid re-visiting same node at same depth
            cache_key = (current_node, depth)
            if cache_key in visited:
                continue
            visited.add(cache_key)

            is_leaf = not dependencies.get(current_node)
            is_deep_enough = depth >= min_depth
            is_too_deep = depth >= max_depth

            if is_leaf or is_too_deep:
                if is_deep_enough:
                    score = score_function(trace)
                    if top_k is None or top_k == math.inf:
                        heapq.heappush(top_traces_heap, (score, tuple(trace)))
                    else:
                        if len(top_traces_heap) < top_k:
                            heapq.heappush(top_traces_heap, (score, tuple(trace)))
                        elif score > top_traces_heap[0][0]:
                            heapq.heappushpop(top_traces_heap, (score, tuple(trace)))
                continue

            extended = False
            for next_node, edge_strength in dependencies[current_node].items():
                if edge_strength > threshold and next_node not in trace:
                    stack.append((next_node, trace + [next_node], depth + 1))
                    extended = True

            if not extended and is_deep_enough:
                score = score_function(trace)
                if len(top_traces_heap) < top_k:
                    heapq.heappush(top_traces_heap, (score, tuple(trace)))
                elif score > top_traces_heap[0][0]:
                    heapq.heappushpop(top_traces_heap, (score, tuple(trace)))

        # Flatten heap into sorted list and add to global list
        sorted_traces = sorted(top_traces_heap, reverse=True)
        all_subtraces.extend([(score, list(trace)) for score, trace in sorted_traces])

    return all_subtraces