from process_fragment_miner.scorer import (
    BigramScorer,
    DependencyScorer,
    SimilarityScorer,
)

class ScorerFactory:
    _registry = {
        "bigram": {
            "class": BigramScorer,
            "from_miner": {"traces": "get_traces"},
        },
        "dependency": {
            "class": DependencyScorer,
            "from_miner": {"dependency_matrix": "dependencies"},
        },
        "similarity": {
            "class": SimilarityScorer,
            "from_miner": {"traces": "get_traces"},
        },
    }

    @classmethod
    def create(cls, name: str, miner=None, **kwargs):
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown scorer: {name}")

        entry = cls._registry[name]
        scorer_cls = entry["class"]
        from_miner = entry.get("from_miner", {})

        init_args = {}
        for param_name, source in from_miner.items():
            if callable(source):
                init_args[param_name] = source(miner)
            elif hasattr(miner, source):
                attr = getattr(miner, source)
                init_args[param_name] = attr() if callable(attr) else attr

        # Allow overrides from kwargs
        init_args.update(kwargs)

        return scorer_cls(**init_args)

    @classmethod
    def register(cls, name, scorer_cls):
        """
        Register a custom scorer class.

        Args:
            name (str): Identifier for the scorer.
            scorer_cls (type): Class inheriting from BaseScorer.
        """
        cls._registry[name.lower()] = scorer_cls