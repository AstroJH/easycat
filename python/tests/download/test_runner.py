"""Offline tests for DownloadRunner with a fake archive."""
from pathlib import Path

import pandas as pd
import pytest

from easycat.download.base import FetchContext, ItemResult, SurveyArchive
from easycat.download.runner import DownloadRunner


class FakeArchive(SurveyArchive):
    name = "fake"
    default_batch_size = 2

    def __init__(self, fail_ids=()):
        super().__init__()
        self.fail_ids = set(fail_ids)
        self.calls = []

    def fetch_batch(self, rows, ctx):
        self.calls.append(list(rows["obj_id"]))
        out = []
        for _, row in rows.iterrows():
            obj_id = str(row["obj_id"])
            if obj_id in self.fail_ids:
                out.append(ItemResult(obj_id=obj_id, success=False, error="boom"))
            else:
                (ctx.store_dir / f"{obj_id}.dat").write_text("ok")
                out.append(ItemResult(obj_id=obj_id, success=True, data=obj_id))
        return out


def make_catalog(n=5):
    return pd.DataFrame({"obj_id": [f"src{i}" for i in range(n)]})


def test_runner_completes(tmp_path):
    arc = FakeArchive()
    summary = DownloadRunner(
        archive=arc, catalog=make_catalog(5), store_dir=tmp_path,
        checkpoint=tmp_path / "cp.json", n_workers=2, progress=False,
    ).run()
    assert summary.total == 5
    assert summary.completed == 5
    assert summary.failed == 0
    assert len(list(tmp_path.glob("*.dat"))) == 5


def test_runner_checkpoint_skip(tmp_path):
    cp = tmp_path / "cp.json"
    arc = FakeArchive()
    r1 = DownloadRunner(archive=arc, catalog=make_catalog(5), store_dir=tmp_path,
                        checkpoint=cp, n_workers=2, progress=False)
    assert r1.run().completed == 5

    arc2 = FakeArchive()
    r2 = DownloadRunner(archive=arc2, catalog=make_catalog(5), store_dir=tmp_path,
                        checkpoint=cp, n_workers=2, progress=False)
    s2 = r2.run()
    assert s2.skipped == 5
    assert arc2.calls == []  # no network work on resume


def test_runner_retries_failed(tmp_path):
    arc = FakeArchive(fail_ids={"src0"})
    r = DownloadRunner(archive=arc, catalog=make_catalog(3), store_dir=tmp_path,
                       checkpoint=tmp_path / "cp.json", n_workers=1,
                       progress=False, max_retries=1)
    summary = r.run()
    assert summary.failed == 1
    assert summary.completed == 2
    assert "src0" in summary.failed_ids


def test_archive_contract_violation_raises(tmp_path):
    class Bad(SurveyArchive):
        name = "bad"
        default_batch_size = 2

        def fetch_batch(self, rows, ctx):
            return [ItemResult(obj_id="x", success=True)]  # too few

    with pytest.raises(RuntimeError):
        DownloadRunner(archive=Bad(), catalog=make_catalog(3), store_dir=tmp_path,
                       checkpoint=None, n_workers=1, progress=False).run()


# --------------------------------------------------------------------------- #
# interruption handling
# --------------------------------------------------------------------------- #
class _SlowArchive(SurveyArchive):
    """Writes one file per source, slowly enough that batches overlap."""

    name = "slow"
    default_batch_size = 1

    def __init__(self, delay=0.05, n_batches=6):
        super().__init__()
        self.delay = delay
        self.n_batches = n_batches

    def fetch_batch(self, rows, ctx):
        import time

        time.sleep(self.delay)
        out = []
        for _, row in rows.iterrows():
            obj_id = str(row["obj_id"])
            (ctx.store_dir / f"{obj_id}.dat").write_text("ok")
            out.append(ItemResult(obj_id=obj_id, success=True, data=obj_id))
        return out


def test_interrupt_records_already_downloaded_sources(tmp_path, monkeypatch):
    """Regression: on Ctrl+C, batches that finished (files already on disk)
    must be recorded in the checkpoint, otherwise the checkpoint under-counts
    and a resumed run re-downloads them.
    """
    import easycat.download.runner as runner_mod
    from easycat.download.checkpoint import CheckpointStore

    real_as_completed = runner_mod.as_completed
    state = {"raised": False}

    def interrupt_after_first(futures):
        for i, fut in enumerate(real_as_completed(futures)):
            yield fut
            # raise once, right after the first batch has been processed,
            # while other batches are still in flight
            if i == 0 and not state["raised"]:
                state["raised"] = True
                raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod, "as_completed", interrupt_after_first)

    catalog = pd.DataFrame({"obj_id": [f"s{i}" for i in range(6)]})
    cp_path = tmp_path / "checkpoint.json"
    runner = DownloadRunner(
        archive=_SlowArchive(),
        catalog=catalog,
        store_dir=tmp_path,
        checkpoint=cp_path,
        n_workers=2,
        progress=False,
    )
    summary = runner.run()

    assert summary.interrupted is True
    on_disk = {p.stem for p in tmp_path.glob("*.dat")}
    assert on_disk, "expected some downloads to have completed"
    assert len(on_disk) > 1, "the test needs several finished batches"

    # every source with a file on disk must be marked done ...
    assert on_disk <= set(summary.failed_ids) | {
        i for i in catalog["obj_id"] if runner.checkpoint.is_done(i)
    }
    # ... and that must be persisted to disk, not only in memory
    reloaded = CheckpointStore(cp_path)
    for obj_id in on_disk:
        assert reloaded.is_done(obj_id), f"{obj_id} downloaded but not recorded"


# --------------------------------------------------------------------------- #
# safety net: files already on disk count as done
# --------------------------------------------------------------------------- #
class _DiskArchive(SurveyArchive):
    """Like FakeArchive but declares its per-source output path."""

    name = "disk"
    default_batch_size = 1

    def __init__(self):
        super().__init__()
        self.calls = []

    def output_path(self, ctx, obj_id):
        return ctx.store_dir / f"{obj_id}.dat"

    def fetch_batch(self, rows, ctx):
        out = []
        for _, row in rows.iterrows():
            obj_id = str(row["obj_id"])
            self.calls.append(obj_id)
            (ctx.store_dir / f"{obj_id}.dat").write_text("ok")
            out.append(ItemResult(obj_id=obj_id, success=True, data=obj_id))
        return out


def test_recover_from_disk_marks_existing_files(tmp_path, caplog):
    """Hard interrupts can leave finished files unrecorded; the runner must
    treat them as done (with a warning) instead of downloading again."""
    import logging
    from easycat.download.checkpoint import CheckpointStore

    # files already on disk for s0/s1, but *not* in the checkpoint
    for obj in ("s0", "s1"):
        (tmp_path / f"{obj}.dat").write_text("ok")

    catalog = pd.DataFrame({"obj_id": ["s0", "s1", "s2"]})
    cp_path = tmp_path / "checkpoint.json"
    archive = _DiskArchive()

    with caplog.at_level(logging.WARNING, logger="easycat.download"):
        runner = DownloadRunner(
            archive=archive, catalog=catalog, store_dir=tmp_path,
            checkpoint=cp_path, n_workers=1, progress=False,
        )
        summary = runner.run()

    assert summary.recovered == 2
    assert set(summary.recovered_ids) == {"s0", "s1"}
    # only the missing source was fetched
    assert archive.calls == ["s2"]
    # recovery is persisted, so a second run skips everything
    reloaded = CheckpointStore(cp_path)
    assert all(reloaded.is_done(i) for i in ("s0", "s1", "s2"))

    warnings = [r.getMessage() for r in caplog.records
                if r.levelname == "WARNING"]
    assert any("missing from the checkpoint" in w for w in warnings), warnings

    # running again does not report any recovery
    again = DownloadRunner(archive=_DiskArchive(), catalog=catalog,
                           store_dir=tmp_path, checkpoint=cp_path,
                           n_workers=1, progress=False).run()
    assert again.recovered == 0 and again.skipped == 3


def test_no_recovery_without_output_path(tmp_path):
    """Archives that do not declare output_path keep the old behaviour."""
    catalog = pd.DataFrame({"obj_id": ["s0"]})
    (tmp_path / "s0.dat").write_text("ok")     # would look "done" if guessed
    archive = FakeArchive()                    # output_path() -> None
    runner = DownloadRunner(archive=archive, catalog=catalog, store_dir=tmp_path,
                            checkpoint=tmp_path / "cp.json", n_workers=1,
                            progress=False)
    summary = runner.run()
    assert summary.recovered == 0
    assert archive.calls == [["s0"]]           # still fetched


class _RowAwareArchive(SurveyArchive):
    """Archive whose path hook needs fields from the catalog row."""

    name = "row-aware"
    default_batch_size = 1

    def __init__(self):
        super().__init__()
        self.calls = []
        self.results = []

    def output_path(self, ctx, obj_id, *, row=None):
        group = str(row["group"]) if row is not None else "unknown"
        return ctx.dest(
            obj_id, ctx.store_dir / group / f"{obj_id}.dat", row=row,
        )

    def fetch_batch(self, rows, ctx):
        out = []
        for _, row in rows.iterrows():
            obj_id = str(row["obj_id"])
            self.calls.append(obj_id)
            path = self.output_path(ctx, obj_id, row=row)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok")
            result = ItemResult(obj_id=obj_id, success=True, data=path)
            out.append(self.enrich_result(result, row=row, dest=path))
        self.results.extend(out)
        return out


def test_runner_passes_row_to_dest_hook_and_enriches_meta(tmp_path):
    catalog = pd.DataFrame({
        "obj_id": ["s0", "s1"],
        "group": ["a", "b"],
    })
    archive = _RowAwareArchive()
    runner = DownloadRunner(
        archive=archive,
        catalog=catalog,
        store_dir=tmp_path,
        checkpoint=tmp_path / "checkpoint.json",
        n_workers=1,
        progress=False,
        dest_fn=lambda obj_id, default, row=None: (
            default.parent / "custom" / default.name
        ),
    )

    summary = runner.run()

    assert summary.completed == 2
    assert (tmp_path / "a" / "custom" / "s0.dat").exists()
    assert (tmp_path / "b" / "custom" / "s1.dat").exists()
    assert archive.results[0].meta["archive"] == "row-aware"
    assert archive.results[0].meta["dest"].endswith("/a/custom/s0.dat")
    assert archive.results[0].meta["size"] == 2


def test_recovery_uses_row_aware_output_path(tmp_path):
    catalog = pd.DataFrame({"obj_id": ["s0"], "group": ["a"]})
    path = tmp_path / "a" / "s0.dat"
    path.parent.mkdir(parents=True)
    path.write_text("already done")

    archive = _RowAwareArchive()
    summary = DownloadRunner(
        archive=archive,
        catalog=catalog,
        store_dir=tmp_path,
        checkpoint=tmp_path / "checkpoint.json",
        n_workers=1,
        progress=False,
    ).run()

    assert summary.recovered == 1
    assert archive.calls == []
