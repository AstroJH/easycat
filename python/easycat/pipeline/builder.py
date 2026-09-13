"""Build linear pipelines from Python dictionaries or YAML files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from .core import Pipeline
from .factory import NodeFactory
import yaml


class PipelineBuilder:
    """Factory for declarative, strictly ordered pipelines.

    YAML describes topology and node parameters only.  Batch concerns such as
    checkpoint paths, worker counts and output collectors belong to
    :class:`easycat.pipeline.runner.PipelineRunner`.
    """

    @staticmethod
    def from_config(config: Dict[str, Any]) -> Pipeline:
        """
        Build pipeline from configuration dictionary.

        Parameters
        ----------
        config : dict
            Mapping with top-level ``pipeline`` and ``nodes`` keys.  Each node
            needs ``class`` and may provide ``name``, ``enabled`` and
            ``params``.

        Returns
        -------
        Pipeline
            Initialized pipeline
        """

        pl_cfg = config.get("pipeline", {})

        pl = Pipeline(
            name=pl_cfg.get("name", "UNNAMED")
        )

        seen = set()
        for node_cfg in config.get("nodes", []):
            if not node_cfg.get("enabled", True):
                continue

            node = NodeFactory.create(node_cfg)
            if node.name in seen:
                raise ValueError(f"duplicate node name in config: {node.name!r}")
            seen.add(node.name)
            pl.add_node(node)

        return pl

    @staticmethod
    def from_yaml(filename: Union[str, Path]) -> Pipeline:
        """Load a YAML pipeline configuration from ``filename``."""
        with open(filename) as f:
            cfg = yaml.safe_load(f)

        return PipelineBuilder.from_config(cfg or {})
