import math
from pathlib import Path
from process_fragment_miner import ProcessFragmentMiner
from pm4py.algo.filtering.log.attributes import attributes_filter
import os

from process_fragment_miner.utils import export_xes_by_fragments, get_fragments_by_labels_from_log, import_xes

def evaluation(logs_dir, export_path, path_filtering=False, dk_fragments=None):
    if os.path.isfile(logs_dir):
        filenames = [os.path.basename(logs_dir)]
        logs_dir = os.path.dirname(logs_dir)
    else:
        filenames = [f for f in os.listdir(logs_dir) if os.path.isfile(os.path.join(logs_dir, f))]

    Path(f'{export_path}/xes').mkdir(parents=True, exist_ok=True)
    
    for filename in filenames:
        event_log = import_xes(logs_dir=logs_dir, filename=filename, path_filtering=path_filtering)
        no_start_end = attributes_filter.apply_events(
            event_log,
            parameters={"attribute_key": "concept:name", "positive": False},
            values=['START', 'END']
        )
        if dk_fragments == None:
            dk_fragments = get_fragments_by_labels_from_log(event_log)
        for fm in ['dk','dependency','bigram','similarity']:
            if fm == "dk":
                fragments = dk_fragments
            else:
                miner = ProcessFragmentMiner(
                    event_log=no_start_end,
                    scorer=fm,
                    # scorer_kwargs={"remove_loops": True}
                )

                # Extract top subtraces using DFS
                subtraces = miner.extract_subtraces(max_depth=1000, min_depth=1, top_k=math.inf)

                # Select the best disjoint subset of fragments
                score, fragments, scores, method_used = miner.mine_best_fragments(
                    subtraces=subtraces,
                    score_agg="sum",       # "sum", "mean", "log_likelihood"
                    alpha=0.0,             # reward per unique activity
                    beam_size=50,
                    max_memory_mb=10000,
                    method="auto" if not filename in ["BPIC15_2f.xes.gz","BPIC15_3f.xes.gz","BPIC15_4f.xes.gz","BPIC15_5f.xes.gz"] else "beam",         # try "dp", "beam", or "auto"
                    return_details=True,
                    ensure_coverage=True
                )

                if fm == 'dependency':
                    fm = 'heuristic'

                if score is not None and scores is not None:
                    txt_path = f'{export_path}/{filename}.pfm.metrics.txt'
                    with open(txt_path, 'a') as f:
                        f.write(f'{fm};{method_used};{score};{fragments}\n')
            print(fm)
            export_xes_by_fragments(event_log,fragments,export_path,filename,fm,include_root=True,plot_root=True,pm4py_metrics=False)
    