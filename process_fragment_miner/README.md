# Process Fragment Miner (PFM) Scripts

This directory contains scripts for processing and mining process fragments as subprocesses.

## Files

process_fragment_miner/
├── miner.py                      # ProcessFragmentMiner class
├── scorer_factory.py             # ScorerFactory class
├── scorer.py                     # Scorer classes (optionally user-defined)
├── fragment_selector.py          # DP + beam logic
├── subtrace_extractor.py         # Top-K DFS subtrace extraction
├── utils.py                      # Minor helpers (RAM usage, bitmask ops)
├── extractors/
│   └── pm4py_adapter.py          # pm4py import + dependency graph
│   └── word2vec_adapter.py.      # Word2Vec adapter for SimilarityScorer

## Usage

Import the required classes or functions in your scripts:

```python
from miner import FragmentMiner
from utils import some_utility_function
```

## Requirements

- Python 3.7+
- See `requirements.txt` for dependencies.

## License

See [LICENSE](../LICENSE) for details.