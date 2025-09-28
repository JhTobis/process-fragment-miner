from collections import defaultdict
from copy import deepcopy
import re
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.algo.evaluation.replay_fitness.variants import token_replay, alignment_based
from pm4py.algo.evaluation.precision.variants import etconformance_token as precision_evaluator
from pm4py.convert import convert_to_bpmn
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.discovery import discover_process_tree_inductive
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.discovery import discover_heuristics_net, discover_process_tree_inductive
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.variants.log.get import get_variants, get_variants_sorted_by_count
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.util import sorting
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.log import converter as df_to_log_converter
import pandas as pd

def merge_event_logs_by_trace_id(log1, log2):
    """
    Merges two PM4Py EventLog objects based on common trace IDs.
    Keeps only common columns, removes NaNs, and ensures 'START' is first in each trace.

    Args:
        log1: First PM4Py EventLog object.
        log2: Second PM4Py EventLog object.

    Returns:
        A merged PM4Py EventLog object.
    """
    # Convert logs to DataFrames
    df1 = log_converter.apply(log1, variant=log_converter.Variants.TO_DATA_FRAME)
    df2 = log_converter.apply(log2, variant=log_converter.Variants.TO_DATA_FRAME)

    # Keep only common columns to avoid NaNs
    common_columns = df1.columns.intersection(df2.columns)
    df1 = df1[common_columns].copy()
    df2 = df2[common_columns].copy()

    # Combine the logs
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Ensure required columns exist
    required = {"case:concept:name", "time:timestamp", "concept:name"}
    if not required.issubset(combined_df.columns):
        raise ValueError(f"Missing one or more required columns: {required}")

    # Force 'START' to be the first event in each trace
    combined_df["start_priority"] = combined_df["concept:name"].apply(lambda x: 0 if x == "START" else 1)
    combined_df.sort_values(by=["case:concept:name", "start_priority", "time:timestamp"], inplace=True)
    combined_df.drop(columns=["start_priority"], inplace=True)

    # Drop rows with any remaining NaNs, just in case
    combined_df.dropna(inplace=True)

    # Convert back to EventLog
    merged_log = df_to_log_converter.apply(combined_df, variant=df_to_log_converter.Variants.TO_EVENT_LOG)

    return merged_log


def calculate_cfc(petri_net):
    cfc = 0
    for place in petri_net.places:
        if len(place.in_arcs) > 1 or len(place.out_arcs) > 1:
            cfc += 1
    for transition in petri_net.transitions:
        if len(transition.in_arcs) > 1 or len(transition.out_arcs) > 1:
            cfc += 1
    return cfc

def calculate_metrics(process_tree, event_log):

    net, im, fm = pt_converter.apply(process_tree)
    log_fitness = pm4py.fitness_alignments(event_log,net, im, fm, multi_processing=True)['log_fitness']
    precision = pm4py.precision_alignments(event_log,net, im, fm, multi_processing=True)

    f1 = 2 * (log_fitness * precision) / (log_fitness + precision)

    cfc_value = calculate_cfc(net)

    size = len(net.places) + len(net.transitions)

    metrics = {
        'fi': log_fitness,
        'pr': precision,
        'F1': f1,
        'CFC': cfc_value,
        'size': size
    }
    return metrics

def calculate_quality_measures(event_log):
    process_tree = inductive_miner.apply(event_log)
    return calculate_metrics(process_tree, event_log)

def calculate_quality_measures_means(fragment_properties):
    sums = defaultdict(float)
    count = 0

    for entry in fragment_properties:
        inner = list(entry.values())[0]
        metrics = inner.get('metrics', {})
        for k, v in metrics.items():
            sums[k] += v
        count += 1

    return {f'{k}_mean': v / count for k, v in sums.items()}

def get_activities(event_log):
    return {event["concept:name"] for trace in event_log for event in trace}

def plot_pm4py_inductive_miner_bpmn(event_log, quality_measures=True):
    process_tree = discover_process_tree_inductive(event_log, noise_threshold= 0.2)

    net, im, fm = pt_converter.apply(process_tree)

    bpmn_model = convert_to_bpmn(process_tree)
    gviz = bpmn_visualizer.apply(bpmn_model)
    bpmn_visualizer.view(gviz)

    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)

    if quality_measures: print(calculate_metrics(process_tree, event_log))

def get_fragments_by_labels(activities):
    grouped = defaultdict(list)
    activities = [a for a in activities if a not in ["START", "END"]]

    for item in activities:
        first_letter = ""
        if item[:2].isdigit():
            parts = item.split("_")
            first_letter = "_".join(parts[:2])
        else:
            first_letter = item[0]
        grouped[first_letter].append(item)

    return list(grouped.values())

def get_fragments_by_labels_from_log(event_log):
    activities = get_activities(event_log)
    return get_fragments_by_labels(activities)

def get_fragment_log(fragments_properties):
    fragment_log = EventLog()
    trace_index = {}

    for fragment in fragments_properties:
        for group_name, events in fragment.items():
            labeled_events = [(e, "start") for e in events.get("start_events", [])] + \
                             [(e, "end") for e in events.get("end_events", [])]

            for entry, label in labeled_events:
                case_id = entry["case_id"]
                event = dict(entry["event"])

                event["concept:name"] = f"{group_name}_{label}"

                if case_id not in trace_index:
                    new_trace = Trace()
                    new_trace.attributes["concept:name"] = case_id
                    trace_index[case_id] = new_trace
                    fragment_log.append(new_trace)

                trace_index[case_id].append(event)
    return sorting.sort_timestamp_log(fragment_log)

def relabel_fragments_in_tree(tree):
    # Helper to traverse in execution order (pre-order)
    def get_nodes_in_order(node):
        nodes = [node]
        for child in node.children:
            nodes.extend(get_nodes_in_order(child))
        return nodes

    all_nodes = get_nodes_in_order(tree)
    
    # Find all start fragment nodes
    fragment_start_nodes = [
        node for node in all_nodes
        if isinstance(node.label, str) and re.match(r"fragment_\d+_start", node.label)
    ]
    
    # Sort by tree execution order (left to right)
    fragment_start_nodes.sort(key=lambda node: all_nodes.index(node))
    
    # Map old fragment numbers to new consistent ones
    fragment_mapping = {}
    for new_id, node in enumerate(fragment_start_nodes, start=0):
        match = re.match(r"fragment_(\d+)_start", node.label)
        old_id = match.group(1)
        fragment_mapping[old_id] = str(new_id)
        node.label = f"fragment_{new_id}_start"
    
    # Update corresponding _end nodes
    for node in all_nodes:
        if isinstance(node.label, str):
            match = re.match(r"fragment_(\d+)_end", node.label)
            if match:
                old_id = match.group(1)
                if old_id in fragment_mapping:
                    new_id = fragment_mapping[old_id]
                    node.label = f"fragment_{new_id}_end"
    
    return tree, fragment_mapping

def relabel_fragments_in_event_log(event_log, fragment_mapping, label_key="concept:name"):
    new_log = deepcopy(event_log)  # avoid modifying original log

    for trace in new_log:
        for event in trace:
            label = event[label_key]
            match = re.match(r"fragment_(\d+)_(start|end)", label)
            if match:
                old_id, position = match.groups()
                if old_id in fragment_mapping:
                    new_id = fragment_mapping[old_id]
                    event[label_key] = f"fragment_{new_id}_{position}"
    
    return new_log

def pm4py_bpmn_heuristic_miner(event_log, event_names, threshold_bpmn=0.0, return_pt=False, plot=True, relabel_fragments=False):
    result = ()
    event_log = attributes_filter.apply_events(
        event_log,
        parameters={"attribute_key": "concept:name","positive": True},
        values=event_names
    )

    process_tree = discover_process_tree_inductive(event_log, noise_threshold=threshold_bpmn)

    if relabel_fragments:
        process_tree, fragment_mapping = relabel_fragments_in_tree(process_tree)
        event_log = relabel_fragments_in_event_log(event_log, fragment_mapping)
        result = (fragment_mapping,)

    bpmn_model = convert_to_bpmn(process_tree)

    if plot:
        gviz = bpmn_visualizer.apply(bpmn_model)
        bpmn_visualizer.view(gviz)
        heu_net = discover_heuristics_net(event_log, dependency_threshold = 0, and_threshold = 0, loop_two_threshold = 0)
        gviz = hn_visualizer.apply(heu_net)
        hn_visualizer.view(gviz)

    if return_pt:
        return (process_tree, event_log, *result)
    return (event_log, *result)

def pm4py_all_fragments(event_log, fragments, fragment_end=True, quality_measures=True, subprocess_threshold_bpmn=0.0, root_process_threshold_bpmn=0.0, plot_fragments=False, plot_root=True):
    fragments_properties = []

    for i, fragment in enumerate(fragments):
        group_name = f'fragment_{i}'

        process_tree, fragment_log = pm4py_bpmn_heuristic_miner(event_log,fragment, threshold_bpmn=subprocess_threshold_bpmn, return_pt=True, plot=plot_fragments)
        
        fragment_properties = {}

        if quality_measures:
            fragment_properties['metrics'] = calculate_metrics(process_tree, fragment_log)
            print(fragment_properties['metrics'])
        
        
        fragment_properties['start_events']= [
                    {
                            "case_id": trace.attributes.get("concept:name"), 
                            "event": min(trace, key=lambda e: e["time:timestamp"])
                    }
                            for trace in fragment_log if len(trace) > 0
                    ]
        fragment_properties['end_events'] = [
                    {
                            "case_id": trace.attributes.get("concept:name"), 
                            "event": max(trace, key=lambda e: e["time:timestamp"])
                    }
                            for trace in fragment_log if len(trace) > 0
                    ] if fragment_end else []

        fragments_properties += [{
            f'{group_name}':fragment_properties
        }]


    start_end_log = attributes_filter.apply_events(
        event_log,
        parameters={"attribute_key": "concept:name","positive": True},
        values=['START','END']
    )

    root_log = get_fragment_log(fragments_properties)
    if len(start_end_log) != 0:
        root_log = merge_event_logs_by_trace_id(root_log,start_end_log)


    process_tree,root_log,fragment_mapping = pm4py_bpmn_heuristic_miner(root_log,list(get_activities(root_log)),threshold_bpmn=root_process_threshold_bpmn, return_pt=True, plot=plot_root, relabel_fragments=True)

    if quality_measures:
        root_model_qm = calculate_metrics(process_tree,root_log)
        mean_qm = calculate_quality_measures_means(fragments_properties)
        print(root_model_qm)
        print(mean_qm)
        return (root_log, root_model_qm, mean_qm,fragment_mapping)
    
    return (root_log,None,None,fragment_mapping)



def export_xes_by_fragments(event_log, fragments, export_path, filename, fm, include_root=True, plot_root=True, plot_fragments=False, pm4py_metrics=False):
    root_event_log,root_model_qm, mean_qm, fragment_mapping = pm4py_all_fragments(event_log,fragments,fragment_end=True,quality_measures=pm4py_metrics,plot_root=plot_root,plot_fragments=plot_fragments)

    if include_root:
        xes_exporter.apply(root_event_log,f'{export_path}/xes/{filename}.root.{fm}.xes.gz')
    
    for i, fragment in enumerate(fragments):
        fragment_event_log = attributes_filter.apply_events(
            event_log,
            parameters={"attribute_key": "concept:name","positive": True},
            values=fragment
        )
        i_relabeled = fragment_mapping[str(i)]
        group_name = f'fragment_{i_relabeled}'
        print(f'\n{group_name}: {fragment}')
        xes_exporter.apply(fragment_event_log,f'{export_path}/xes/{filename}.{i_relabeled}.{fm}.xes.gz')
    
    if root_model_qm is not None and mean_qm is not None:
        txt_path = f'{export_path}/{filename}.pm4py.metrics.txt'
        with open(txt_path, 'a') as f:
            f.write(f'{fm};{root_model_qm};{mean_qm}\n')

def import_xes(logs_dir, filename, path_filtering=False):
    full_event_log = xes_importer.apply(f'{logs_dir}/{filename}')

    if path_filtering:
        variants = get_variants(full_event_log)

        variants_count = get_variants_sorted_by_count(variants)

        # Total number of cases in log
        total_cases = sum(variant[1] for variant in variants_count)

        # Define the threshold (e.g., keep variants that together cover 80%)
        threshold = 0.8
        cumulative = 0
        selected_variants = []

        for variant in variants_count:
            cumulative += variant[1]
            selected_variants.append(variant[0])
            if cumulative / total_cases >= threshold:
                break

        # Filter log by selected (frequent) variants
        full_event_log = variants_filter.apply(full_event_log, selected_variants)
    
    return full_event_log