from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.filtering.log.attributes import attributes_filter


def load_event_log(file_path):
    """
    Loads an event log using PM4Py from a .xes file.

    Args:
        file_path (str): Path to a .xes log file.

    Returns:
        event_log (pm4py log object)
    """
    return xes_importer.apply(file_path)


def extract_dependency_graph(event_log, threshold=0.2):
    """
    Extracts a dependency graph from an event log using the Heuristics Miner.

    Args:
        event_log: The PM4Py event log object.
        threshold (float): Minimum dependency value to include an edge.

    Returns:
        dict: {activity: {next_activity: strength, ...}}
    """
    # Apply Heuristics Miner to extract the dependency matrix
    heu_net = heuristics_miner.apply_heu(event_log)
    dependency_matrix = heu_net.dependency_matrix

    graph = {}

    for act_from in dependency_matrix:
        for act_to in dependency_matrix[act_from]:
            strength = dependency_matrix[act_from][act_to]
            if strength >= threshold:
                if act_from not in graph:
                    graph[act_from] = {}
                graph[act_from][act_to] = strength

    return graph