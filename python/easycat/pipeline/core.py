"""Core objects for the easycat processing pipeline."""
from __future__ import annotations

import copy
import logging
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("easycat")


@dataclass
class DataPacket:
    """Container passed between processing nodes.

    ``light_curve`` and the legacy ``results``/``errors``/``warnings`` fields
    remain available.  New code should use ``obj_id``, ``artifacts`` and
    ``node_outcomes`` for batch execution and provenance.
    """

    # Primary scientific payload.  Current nodes use a pandas light curve.
    light_curve: Optional[pd.DataFrame] = None
    # Per-run inputs such as file paths, position references and output paths.
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Legacy metrics format: results["<key>@<node>"] = {...}.
    results: Dict[str, Any] = field(default_factory=dict)
    # Append-only diagnostics.  Node status is based on per-node deltas,
    # not on the presence of historical errors in this list.
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    # Batch identity and provenance used by PipelineRunner.
    obj_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    # Last outcome for each node in the current pipeline run.
    node_outcomes: Dict[str, "NodeOutcome"] = field(default_factory=dict)

    def clone(self) -> "DataPacket":
        """Return an independent packet suitable for a worker/task."""
        # DataFrames are copied explicitly; metadata/provenance are deep
        # copied so worker tasks cannot mutate another source's state.
        return DataPacket(
            light_curve=None if self.light_curve is None else self.light_curve.copy(),
            metadata=copy.deepcopy(self.metadata),
            results=copy.deepcopy(self.results),
            errors=copy.deepcopy(self.errors),
            warnings=copy.deepcopy(self.warnings),
            obj_id=self.obj_id,
            provenance=copy.deepcopy(self.provenance),
            artifacts=copy.deepcopy(self.artifacts),
            node_outcomes=dict(self.node_outcomes),
        )

    def add_result(
        self,
        key: str,
        value: Any,
        node_name: Optional[str] = None,
    ) -> None:
        """Store a processing result without changing the legacy schema."""
        result_entry = {
            "value": value,
            "timestamp": datetime.now(),
            "node": node_name,
        }
        # Keep the historical key format for backwards compatibility.
        self.results[f"{key}@{node_name}"] = result_entry

    def result_map(self, node_name: str) -> Dict[str, Any]:
        """Return a node's metrics as a plain ``{key: value}`` mapping."""
        suffix = f"@{node_name}"
        return {
            key[: -len(suffix)]: value.get("value")
            for key, value in self.results.items()
            if key.endswith(suffix)
        }

    def add_error(
        self,
        message: str,
        node_name: str,
        *,
        exception: Optional[BaseException] = None,
    ) -> None:
        """Record an error message."""
        self.errors.append({
            "message": message,
            "node": node_name,
            "timestamp": datetime.now(),
            "exception": None if exception is None else repr(exception),
        })

    def add_warning(self, message: str, node_name: Optional[str] = None) -> None:
        """Record a warning message."""
        self.warnings.append({
            "message": message,
            "node": node_name,
            "timestamp": datetime.now(),
        })

    def get_result_value(self, key: str, node_name: str) -> Any:
        """Return one legacy result value, or ``None`` if absent."""
        entry = self.results.get(f"{key}@{node_name}")
        return None if entry is None else entry.get("value")


class NodeStatus(Enum):
    """Processing state of a node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeOutcome:
    """Outcome of one node execution."""

    node_name: str
    status: NodeStatus = NodeStatus.PENDING
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    elapsed: Optional[float] = None
    input_rows: Optional[int] = None
    output_rows: Optional[int] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status is NodeStatus.COMPLETED


class ProcessingNode(ABC):
    """Base class for one deterministic processing step."""

    def __init__(
        self,
        name: str,
        minsize: int = 1,
        *,
        required_columns: Optional[Iterable[str]] = None,
        required_metadata: Optional[Iterable[str]] = None,
    ):
        self.name = name
        self.status = NodeStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.execution_time: Optional[float] = None
        self.config: Dict[str, Any] = {}
        self.minsize = minsize
        self.required_columns = tuple(required_columns or ())
        self.required_metadata = tuple(required_metadata or ())

    @property
    def last_outcome(self) -> Optional[NodeOutcome]:
        return getattr(self, "_last_outcome", None)

    @abstractmethod
    def process(self, data: DataPacket) -> DataPacket:
        """Execute the node transformation."""

    def validate(self, data: DataPacket) -> Tuple[bool, Optional[str]]:
        """Validate input without creating groups/detectors or mutating state."""
        if self.minsize >= 1:
            if data.light_curve is None:
                return False, "No light curve data available"
            if len(data.light_curve) < self.minsize:
                return False, "Light curve data is empty"
        if self.required_columns and data.light_curve is not None:
            missing = [
                col for col in self.required_columns
                if col not in data.light_curve.columns
            ]
            if missing:
                return False, f"Missing required columns: {missing}"
        if self.required_metadata:
            missing_metadata = [
                key for key in self.required_metadata
                if data.metadata.get(key) is None
            ]
            if missing_metadata:
                return False, f"Missing required metadata: {missing_metadata}"
        return True, None

    def execute(self, data: DataPacket) -> DataPacket:
        """Execute with isolated per-node error/state accounting."""
        self.start_time = datetime.now()
        self.status = NodeStatus.RUNNING
        # Errors and warnings are accumulated on the packet across the whole
        # pipeline.  Record the current lengths so this node can be judged
        # only on diagnostics produced during this execution.
        error_start = len(data.errors)
        warning_start = len(data.warnings)
        input_rows = None if data.light_curve is None else len(data.light_curve)
        outcome = NodeOutcome(
            node_name=self.name,
            status=NodeStatus.RUNNING,
            started_at=self.start_time,
            input_rows=input_rows,
        )
        data.node_outcomes[self.name] = outcome
        self._last_outcome = outcome

        logger.info("Starting node: %s", self.name)
        try:
            # Validation is intentionally side-effect free.  Nodes create
            # expensive helpers such as groupers only inside process().
            is_valid, error_msg = self.validate(data)
            if not is_valid:
                data.add_error(error_msg or "validation failed", self.name)
            else:
                data = self.process(data)
        except Exception as exc:  # noqa: BLE001 - preserve node-level failure
            data.add_error(traceback.format_exc(), self.name, exception=exc)
            logger.error("Node %s failed: %s", self.name, exc)

        self.end_time = datetime.now()
        self.execution_time = (self.end_time - self.start_time).total_seconds()
        # Handle both in-place mutation and a replacement DataPacket returned
        # by a custom process() implementation.
        new_errors = data.errors[error_start:]
        new_warnings = data.warnings[warning_start:]
        self.status = NodeStatus.FAILED if new_errors else NodeStatus.COMPLETED
        outcome.status = self.status
        outcome.ended_at = self.end_time
        outcome.elapsed = self.execution_time
        outcome.output_rows = None if data.light_curve is None else len(data.light_curve)
        outcome.errors = list(new_errors)
        outcome.warnings = list(new_warnings)
        data.node_outcomes[self.name] = outcome
        return data

    def reset(self) -> None:
        """Reset node execution state."""
        self.status = NodeStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.execution_time = None
        self._last_outcome = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


@dataclass
class PipelineRun:
    """Metadata for one pipeline invocation."""

    run_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    obj_id: Optional[str] = None
    succeeded: bool = False
    failed_nodes: List[str] = field(default_factory=list)


class Pipeline:
    """Execute a sequence of processing nodes."""

    def __init__(self, name: str):
        self.name = name
        self.nodes: List[ProcessingNode] = []
        self.pipeline_start_time: Optional[datetime] = None
        self.pipeline_end_time: Optional[datetime] = None
        self.execution_count: int = 0
        self.last_run: Optional[PipelineRun] = None

    def add_node(self, node: ProcessingNode) -> "Pipeline":
        if any(existing.name == node.name for existing in self.nodes):
            raise ValueError(f"duplicate node name: {node.name!r}")
        self.nodes.append(node)
        logger.debug("Added node %r to pipeline %r", node.name, self.name)
        return self

    def add_nodes(self, nodes: List[ProcessingNode]) -> "Pipeline":
        for node in nodes:
            self.add_node(node)
        return self

    def remove_node(self, name: str) -> bool:
        for i, node in enumerate(self.nodes):
            if node.name == name:
                self.nodes.pop(i)
                return True
        return False

    def validate(self, data: DataPacket) -> List[Tuple[str, str]]:
        """Run all validators without executing nodes or changing node state."""
        failures: List[Tuple[str, str]] = []
        for node in self.nodes:
            valid, message = node.validate(data)
            if not valid:
                failures.append((node.name, message or "validation failed"))
        return failures

    def run(self, data: DataPacket, break_on_error: bool = True) -> DataPacket:
        """Execute all nodes in order on one packet."""
        run = PipelineRun(
            run_id=uuid.uuid4().hex,
            started_at=datetime.now(),
            obj_id=data.obj_id,
        )
        self.last_run = run
        self.pipeline_start_time = run.started_at
        self.execution_count += 1
        # Outcomes describe the current run only.  Historical errors remain
        # on the packet, but node execute() compares error indices so stale
        # errors cannot poison a rerun.
        data.node_outcomes.clear()

        for node in self.nodes:
            node.reset()

        logger.info("Starting pipeline %r with %d nodes", self.name, len(self.nodes))
        for i, node in enumerate(self.nodes, 1):
            data = node.execute(data)
            status = "FAILED" if node.status is NodeStatus.FAILED else "COMPLETED"
            logger.info("[%d/%d] %s: %s", i, len(self.nodes), node.name, status)
            if node.status is NodeStatus.FAILED:
                run.failed_nodes.append(node.name)
                if break_on_error:
                    # Circuit breaker: the rest of this source's pipeline is
                    # intentionally skipped on the first failed node.
                    logger.info("Pipeline stopped at node %r", node.name)
                    break

        run.ended_at = datetime.now()
        run.succeeded = not run.failed_nodes
        self.pipeline_end_time = run.ended_at
        data.provenance.setdefault("pipeline_runs", []).append({
            "name": self.name,
            "run_id": run.run_id,
            "started_at": run.started_at.isoformat(),
            "ended_at": run.ended_at.isoformat(),
            "succeeded": run.succeeded,
            "failed_nodes": list(run.failed_nodes),
        })
        return data

    def get_failed_nodes(self) -> List[str]:
        """Get failed nodes from the last execution."""
        return [] if self.last_run is None else list(self.last_run.failed_nodes)


__all__ = [
    "DataPacket",
    "NodeOutcome",
    "NodeStatus",
    "Pipeline",
    "PipelineRun",
    "ProcessingNode",
]
