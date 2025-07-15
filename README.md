# ProcessFragmentMiner (PFM)

This directory contains scripts for processing and mining process fragments as subprocesses.

## Files

process_fragment_miner/
├── miner.py                # ProcessFragmentMiner class
├── scorer_factory.py       # ScorerFactory class
├── scorer.py               # Scorer classes (optionally user-defined)
├── fragment_selector.py    # DP + beam logic
├── subtrace_extractor.py   # Top-K DFS subtrace extraction
├── utils.py                # Minor helpers (RAM usage, bitmask ops)
├── adapters/
│   ├── pm4py_adapter.py    # pm4py import + dependency graph
│   └── word2vec_adapter.py # Word2Vec adapter for SimilarityScorer

## Usage

Import the required classes or functions in your scripts:

```python
from miner import FragmentMiner
from utils import some_utility_function
```

## Requirements

- Python 3.11
- See [pyproject.toml](pyproject.toml)for dependencies.

## License

This software is licensed under the [GNU Affero General Public License v3 (AGPL-3.0)](LICENSE).
Commercial use is **not permitted without prior written permission** from the author.

Contact for licensing: [joern.tobis@tum.de](mailto:joern.tobis@tum.de)

## Citation

If you use this code in your work, please cite the corresponding publication.

## Disclaimer

See [DISCLAIMER.md](DISCLAIMER.md) — this software is provided **"as is"**, with **no warranty**.