from typing import Dict
from .core import Pipeline
from .factory import NodeFactory
import yaml

class PipelineBuilder:
    @staticmethod
    def from_config(config: Dict) -> Pipeline:
        """
        Build pipeline from configuration dictionary.

        Parameters
        ----------
        config : dict
            Pipeline configuration

        Returns
        -------
        Pipeline
            Initialized pipeline
        """

        pl_cfg = config.get("pipeline", {})

        pl = Pipeline(
            name=pl_cfg.get("name", "UNNAMED")
        )

        for node_cfg in config.get("nodes", []):
            if not node_cfg.get("enabled", True):
                continue

            node = NodeFactory.create(node_cfg)
            pl.add_node(node)

        return pl

    @staticmethod
    def from_yaml(filename):
        with open(filename) as f:
            cfg = yaml.safe_load(f)

        return PipelineBuilder.from_config(cfg)