import pytest
from pydantic import ValidationError

from pfb_app.domain.models import (
    BehaviorEvent,
    BehaviorOutput,
    EvidenceItem,
    IdentityOutput,
    Judgment,
    PfbOutput,
    StageRun,
)


def test_pfb_output_rejects_generated_without_text():
    with pytest.raises(ValidationError):
        PfbOutput(pfb_status="generated", pfb=None, evidence_refs=[])
    with pytest.raises(ValidationError):
        PfbOutput(pfb_status="generated", pfb="  ", evidence_refs=["a"])


def test_pfb_output_rejects_insufficient_with_text():
    with pytest.raises(ValidationError):
        PfbOutput(pfb_status="insufficient_positive_evidence", pfb="text", evidence_refs=[])
    with pytest.raises(ValidationError):
        PfbOutput(pfb_status="insufficient_positive_evidence", pfb=None, evidence_refs=["a"])


def test_pfb_output_valid():
    assert PfbOutput(pfb_status="generated", pfb="良い", evidence_refs=["a"]).pfb == "良い"
    assert PfbOutput(pfb_status="insufficient_positive_evidence", pfb=None, evidence_refs=[]).pfb is None


def test_judgment_extra_field_forbidden():
    with pytest.raises(ValidationError):
        Judgment(ref_id="a", contribution_interpretation="x", is_positive_contribution=True, bogus=1)
    j = Judgment(ref_id="a", contribution_interpretation="x", is_positive_contribution=True)
    assert j.rubric_criteria == []


def test_behavior_and_identity_schemas():
    ev = {"video_time": "00:05", "participant": "Pa", "observed_behavior": "b",
          "evidence_source": "video", "confidence": "high"}
    out = BehaviorOutput(events=[ev])
    assert out.events[0].utterance == "" and out.events[0].speaker == "unknown"
    with pytest.raises(ValidationError):
        BehaviorOutput(events=[{**ev, "evidence_source": "other"}])
    assert IdentityOutput(participants=[]).participants == []


def test_stored_models_extend_drafts():
    e = BehaviorEvent(event_id="cam1-0001", camera="cam1", task_elapsed="00:01:00",
                      video_time="00:05", participant="Pa", observed_behavior="b",
                      evidence_source="audio", confidence="low")
    assert e.event_id == "cam1-0001"
    i = EvidenceItem(item_id="p3-001", quote_verified=True, page=3, quote="q",
                     observed_content="o", evidence_source="text", confidence="medium")
    assert i.page == 3


def test_stage_run_roundtrip():
    r = StageRun(id=None, submission_id=1, stage="pfb", camera=None, subject_id=2, version=1,
                 source="model", upstream_run_ids=[1], prompt_hash=None, model_id=None,
                 output={"a": 1}, status="ok", error=None, warnings=[], duration_sec=1.5,
                 created_at="2026-10-20T00:00:00")
    assert r.stage.value == "pfb"
