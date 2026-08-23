from typing import Dict, Type
import importlib
import yaml
import inspect
import logging

from .core import ProcessingNode

logger = logging.getLogger('easycat')

NODE_REGISTRY: Dict[str, Type[ProcessingNode]] = {}

def register_node(name: str):
    def decorator(cls):
        if name in NODE_REGISTRY:
            raise ValueError(
                f"Node already registered: {name}"
            )

        NODE_REGISTRY[name] = cls

        return cls
    return decorator

def get_node(name: str):

    if name not in NODE_REGISTRY:
        raise KeyError(
            f"Unknown node: {name}. "
            f"Available: {list(NODE_REGISTRY)}"
        )

    return NODE_REGISTRY[name]


class NodeFactory:

    @staticmethod
    def create(node_cfg: dict) -> ProcessingNode:

        NodeClass = NodeFactory.resolve_class(
            node_cfg["class"]
        )


        node_name = node_cfg.get(
            "name",
            NodeClass.__name__
        )


        params = node_cfg.get("params", {})

        params = NodeFactory.validate_params(
            NodeClass,
            params
        )

        return NodeClass(
            name=node_name,
            **params
        )


    @staticmethod
    def resolve_class(name: str):
        """
        Resolve Node class.

        Resolution priority:
        1. Registered node name (names cannot contain '.')
        2. Relative import path starting with '.' (relative to ``easycat.pipeline``)
        2. Absolute Python import path
        """

        # Try registered node first
        if '.' not in name:
            try:
                return get_node(name)
            except KeyError:
                raise ValueError(
                    f"Unknown registered node '{name}'."
                )

        # A leading dot means the prefix ``easycat.pipeline`` is omitted.
        if name.startswith('.'):
            name = f"easycat.pipeline{name}"

        # Resolve Python import path.
        module_name, cls_name = name.rsplit(".", 1)

        try:

            module = importlib.import_module(module_name)
            NodeClass = getattr(module, cls_name)

        except (
            ImportError,
            AttributeError
        ) as e:
            
            raise ImportError(
                f"Cannot load node class: {name}"
            ) from e

        # Check inheritance
        if not issubclass(
            NodeClass,
            ProcessingNode
        ):
            raise TypeError(
                f"{name} is not a ProcessingNode"
            )

        return NodeClass


    @staticmethod
    def validate_params(
        NodeClass,
        params: Dict
    ):

        signature = inspect.signature(
            NodeClass.__init__
        )

        parameters = signature.parameters

        accepted = set()
        accepts_kwargs = False
        missing = []

        param_iter = iter(parameters.items())
        next(param_iter, None) # Skip the instance parameter

        for name, parameter in param_iter:
            if name == "name":
                continue

            # IMPORTANT: Skip **kwargs
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_kwargs = True
                continue

            accepted.add(name)

            if (
                parameter.default is inspect.Parameter.empty
                and name not in params
            ):
                missing.append(name)

        if missing:
            raise ValueError(
                f"Missing required parameters "
                f"{missing} for {NodeClass.__name__}"
            )

        valid = {}

        # Filter parameters
        for key, value in params.items():
            if key in accepted or accepts_kwargs:
                valid[key] = value
            else:
                logger.warning(
                    "Ignoring unknown parameter '%s' "
                    "for node %s",
                    key,
                    NodeClass.__name__
                )

        return valid
