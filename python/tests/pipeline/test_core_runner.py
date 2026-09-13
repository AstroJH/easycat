"""Tests for pipeline state semantics and batch execution."""
from __future__ import annotations

import pandas as pd
import pytest

from easycat.pipeline import (
    DataPacket,
    NodeStatus,
    Pipeline,
    PipelineRunner,
    ProcessingNode,
)


class _AddNode(ProcessingNode):
    def __init__(self, name="add", required=("mjd",)):
        super().__init__(name, required_columns=required)

    def process(self, data):
        data.light_curve = data.light_curve.copy()
        data.light_curve["value"] = 1
        data.add_result("rows", len(data.light_curve), self.name)
        return data


class _FailNode(ProcessingNode):
    def process(self, data):
        data.add_error("intentional", self.name)
        return data


def _process_pipeline_factory():
    return Pipeline("process").add_node(_AddNode())


def _process_resolver(row):
    return DataPacket(light_curve=pd.DataFrame({"mjd": [1.0, 2.0]}))


def _process_collector(packet):
    return {"rows": len(packet.light_curve)}


def test_node_errors_are_isolated_to_current_execution():
    packet = DataPacket(light_curve=pd.DataFrame({"mjd": [1.0, 2.0]}))
    pipeline = Pipeline("state")
    fail = _FailNode("fail")
    add = _AddNode("add")
    pipeline.add_nodes([fail, add])

    result = pipeline.run(packet, break_on_error=False)

    assert fail.status is NodeStatus.FAILED
    assert add.status is NodeStatus.COMPLETED
    assert result.node_outcomes["add"].success
    assert result.get_result_value("rows", "add") == 2
    assert len(result.errors) == 1


def test_circuit_breaker_is_default():
    packet = DataPacket(light_curve=pd.DataFrame({"mjd": [1.0, 2.0]}))
    pipeline = Pipeline("stop").add_nodes([_FailNode("fail"), _AddNode("add")])
    result = pipeline.run(packet)

    assert pipeline.get_failed_nodes() == ["fail"]
    assert result.node_outcomes["fail"].status is NodeStatus.FAILED
    assert "add" not in result.node_outcomes


def test_validate_required_columns_is_non_mutating():
    class _SideEffectNode(ProcessingNode):
        def __init__(self):
            super().__init__("side", required_columns=("required",))
            self.created = False

        def validate(self, data):
            ok, msg = super().validate(data)
            if ok:
                # No state should be needed; this flag checks validate itself.
                self.created = True
            return ok, msg

        def process(self, data):
            return data

    node = _SideEffectNode()
    ok, _ = node.validate(DataPacket(light_curve=pd.DataFrame({"required": [1]})))
    assert ok
    broken = node.validate(DataPacket(light_curve=pd.DataFrame({"other": [1]})))
    assert broken[0] is False
    assert "required" in broken[1]


def test_packet_clone_is_independent():
    packet = DataPacket(
        light_curve=pd.DataFrame({"mjd": [1.0]}),
        metadata={"a": {"b": 1}},
    )
    clone = packet.clone()
    clone.light_curve.loc[0, "mjd"] = 9
    clone.metadata["a"]["b"] = 2
    assert packet.light_curve.loc[0, "mjd"] == 1
    assert packet.metadata["a"]["b"] == 1


def test_pipeline_runner_checkpoint_and_collector(tmp_path):
    catalog = pd.DataFrame({"obj_id": ["a", "b", "c"]})

    def resolver(row):
        return DataPacket(
            obj_id=row.obj_id,
            light_curve=pd.DataFrame({"mjd": [1.0, 2.0], "x": [1, 2]}),
        )

    def factory():
        return Pipeline("batch").add_node(_AddNode())

    def collector(packet):
        return {"rows": len(packet.light_curve), "value": int(packet.light_curve.value.iloc[0])}

    checkpoint = tmp_path / "checkpoint.json"
    summary = PipelineRunner(
        factory, catalog, resolver, collector=collector,
        checkpoint=checkpoint, progress=False, n_workers=2,
    ).run()

    assert summary.completed == 3 and summary.failed == 0
    assert set(summary.results_frame()["obj_id"]) == {"a", "b", "c"}

    again = PipelineRunner(
        factory, catalog, resolver, collector=collector,
        checkpoint=checkpoint, progress=False, n_workers=2,
    ).run()
    assert again.skipped == 3 and again.completed == 3
    assert len(again.results_frame()) == 3


def test_runner_retries_failed(tmp_path):
    catalog = pd.DataFrame({"obj_id": ["a"]})
    calls = {"n": 0}

    class Flaky(ProcessingNode):
        def process(self, data):
            calls["n"] += 1
            if calls["n"] == 1:
                data.add_error("first failure", self.name)
            return data

    summary = PipelineRunner(
        lambda: Pipeline("flaky").add_node(Flaky("flaky")),
        catalog,
        lambda row: DataPacket(light_curve=pd.DataFrame({"x": [1]})),
        checkpoint=tmp_path / "cp.json",
        progress=False,
        retry_failed=True,
        max_retries=1,
    ).run()
    assert summary.completed == 1 and summary.failed == 0


def test_summary_serializes_nested_metrics(tmp_path):
    summary = PipelineRunner(
        lambda: Pipeline("nested").add_node(_AddNode()),
        pd.DataFrame({"obj_id": ["a"]}),
        lambda row: DataPacket(light_curve=pd.DataFrame({"mjd": [1.0]})),
        collector=lambda packet: {"nested": {"a": [1, 2]}},
        checkpoint=tmp_path / "cp.json",
        progress=False,
    ).run()
    frame = summary.results_frame()
    assert isinstance(frame.loc[0, "nested"], str)
    assert '"a": [1, 2]' in frame.loc[0, "nested"]


def test_runner_process_mode(tmp_path):
    summary = PipelineRunner(
        _process_pipeline_factory,
        pd.DataFrame({"obj_id": ["a"]}),
        _process_resolver,
        collector=_process_collector,
        checkpoint=tmp_path / "cp.json",
        progress=False,
        n_workers=1,
        mode="process",
    ).run()
    assert summary.completed == 1 and summary.failed == 0


def test_runner_process_mode_supports_closures(tmp_path):
    offset = 7

    def resolver(row):
        return DataPacket(
            light_curve=pd.DataFrame({"mjd": [1.0, 2.0]}),
            metadata={"offset": offset},
        )

    def collector(packet):
        return {"offset": packet.metadata["offset"], "rows": len(packet.light_curve)}

    summary = PipelineRunner(
        _process_pipeline_factory,
        pd.DataFrame({"obj_id": ["closure"]}),
        resolver,
        collector=collector,
        checkpoint=tmp_path / "closure.json",
        progress=False,
        n_workers=1,
        mode="process",
    ).run()

    assert summary.completed == 1
    assert summary.results_frame().loc[0, "offset"] == 7
