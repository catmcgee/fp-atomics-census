# GPU pod lifetime and evidence

The recovered C2 pods were still running after their recorded provider termination
deadlines. New experiments need a provider timer, a separate local watcher, and
explicit verification that the exact campaign pod was removed.

Use a new `census-` pod name. Before creating it, reserve time for downloads,
compilation, evidence transfer and teardown within the campaign spending cap.
Save the create response and actual quoted rate immediately in a durable campaign
working directory. Do not rely on `/tmp` as the sole copy of evidence.

Set the provider's `--terminate-after` deadline when creating the pod. The installed
runpodctl 2.6.1 expects an ISO datetime, as verified in its help and
[source at 32e9aec](https://github.com/runpod/runpodctl/blob/32e9aec/cmd/pod/create.go).
Check the returned pod configuration; a requested timer is not proof it is active.
This investigation did not establish why the earlier timers failed.

Immediately after creation, launch the local watcher independently of the agent
and GPU queue. Substitute the exact id, name, deadline and durable log paths:

```sh
nohup .venv/bin/python probes/ops/watch_pod.py \
  --pod-id POD_ID --pod-name census-EXPERIMENT \
  --deadline-utc YYYY-MM-DDTHH:MM:SSZ \
  --log /absolute/campaign/path/watchdog.jsonl \
  > /absolute/campaign/path/watchdog-process.log 2>&1 < /dev/null &
```

Check the watcher process and its `started`/`waiting` entries. The deadline should
allow the queue to stop and evidence to sync before the watcher removes the pod.
The watcher removes only the explicitly supplied id with the exact expected name.
It includes EXITED pods, which can retain billable storage. It verifies absence
in two consecutive `pod list --all` responses; a successful delete response alone
is not sufficient.
Identity mismatches and invalid list responses abort without deleting anything.
Provider failures or a pod still listed three minutes after deletion attempts
produce a failure log and a nonzero exit code.

The local watcher needs this machine awake, network access and working runpodctl
credentials. It cannot impose a provider-enforced dollar cap or guarantee removal
if those fail. Keep the provider timer too and monitor actual spend. Do not put
the account API key on a rented pod solely to implement this watcher.

Sync and hash-check completed arms after each run. At normal completion, sync
before deleting the campaign pod, then verify its absence explicitly. The watcher
can also observe that early removal. Leave unrelated account resources alone.
Missing evidence or interrupted jobs must remain labelled missing; do not infer
success from the absence of an error message.

CPU tests use a fake provider and never create or delete a real resource:

```sh
.venv/bin/python -m pytest -q tests/test_pod_watchdog.py
```
