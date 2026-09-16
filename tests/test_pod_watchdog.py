"""Provider lifecycle checks must never delete an unrelated pod or fake success."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "watch_pod", Path(__file__).parents[1] / "probes/ops/watch_pod.py")
watch_pod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch_pod)


def test_deadline_removes_only_target_and_verifies_absence():
    clock = [0]
    calls, events = [], []
    pods = [{"id": "ours", "name": "census-test", "desiredStatus": "EXITED"},
            {"id": "other", "name": "other-project", "desiredStatus": "RUNNING"}]

    def command(*args):
        calls.append((clock[0], args))
        if args[1] == "delete":
            assert args[2] == "ours"
            pods.pop(0)
            return ""
        return json.dumps(pods)

    watch_pod.watch("ours", "census-test", 20, 15,
                   lambda event, **kw: events.append(event), command=command,
                   now=lambda: clock[0], sleep=lambda n: clock.__setitem__(0, clock[0]+n))
    assert [(t, c) for t, c in calls if c[1] == "delete"] == [(20, ("pod", "delete", "ours"))]
    assert events[-1] == "absence_verified"
    assert pods == [{"id": "other", "name": "other-project", "desiredStatus": "RUNNING"}]


@pytest.mark.parametrize("payload", [
    [{"id": "ours", "name": "other-project"}],
    [{"id": "ours", "name": "census-test"}] * 2,
    {}, [{"name": "census-test"}],
])
def test_invalid_identity_never_deletes(payload):
    calls = []
    def command(*args):
        calls.append(args)
        return json.dumps(payload)
    with pytest.raises(ValueError):
        watch_pod.watch("ours", "census-test", 0, 1, lambda *a, **k: None,
                       command=command, now=lambda: 100)
    assert calls == [("pod", "list", "--all")]


def test_delete_response_is_not_proof_of_removal():
    clock = [0]
    events = []
    def command(*args):
        return json.dumps([{"id": "ours", "name": "census-test"}])
    with pytest.raises(RuntimeError, match="still listed"):
        watch_pod.watch("ours", "census-test", 0, 60,
                       lambda event, **kw: events.append(event), command=command,
                       now=lambda: clock[0], sleep=lambda n: clock.__setitem__(0, clock[0]+n))
    assert "absence_verified" not in events


def test_transient_list_omission_does_not_end_watch():
    clock = [0]
    replies = iter([[], [{"id": "ours", "name": "census-test"}], [], []])
    calls, events = [], []
    def command(*args):
        calls.append(args)
        return json.dumps(next(replies)) if args[1] == "list" else ""
    watch_pod.watch("ours", "census-test", 0, 1,
                   lambda event, **kw: events.append(event), command=command,
                   now=lambda: clock[0], sleep=lambda n: clock.__setitem__(0, clock[0]+n))
    assert ("pod", "delete", "ours") in calls
    assert events.count("absence_pending_confirmation") == 2
    assert events[-1] == "absence_verified"
