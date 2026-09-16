#!/usr/bin/env python3
"""Remove one explicitly named census pod at a deadline and verify its absence.

Uses local runpodctl credentials. Run independently of the experiment process;
this is a second guard, not a replacement for the provider's termination timer.
"""
import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path


def utc_deadline(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("deadline must include a UTC offset")
    return parsed.timestamp()


def target_pod(pods, pod_id, name):
    if not isinstance(pods, list):
        raise ValueError("provider pod list is not a list")
    if any(not isinstance(pod, dict) or not pod.get("id") for pod in pods):
        raise ValueError("provider pod list contains an invalid entry")
    matches = [pod for pod in pods if pod["id"] == pod_id]
    if len(matches) > 1:
        raise ValueError("provider returned duplicate pod ids")
    if not matches:
        return None
    if matches[0].get("name") != name:
        raise ValueError("pod name differs from the explicitly authorised target")
    return matches[0]


def cli(*args):
    result = subprocess.run(["runpodctl", *args], check=True, capture_output=True,
                            text=True, timeout=30)
    return result.stdout


def watch(pod_id, name, deadline, poll_seconds, log, *, command=cli,
          now=time.time, sleep=time.sleep):
    # Do not infer ownership from the account or delete every exited pod.
    if not name.startswith("census-"):
        raise ValueError("expected an explicit census- pod name")
    attempted_at = None
    consecutive_errors = 0
    absent_reads = 0
    while True:
        try:
            pods = json.loads(command("pod", "list", "--all"))
            pod = target_pod(pods, pod_id, name)
        except ValueError:
            # Invalid identity/list data is not permission to terminate anything.
            raise
        except (subprocess.SubprocessError, OSError) as error:
            consecutive_errors += 1
            absent_reads = 0
            log("provider_read_failed", error_type=type(error).__name__,
                consecutive_errors=consecutive_errors)
            if consecutive_errors >= 5:
                raise RuntimeError("cannot verify target pod after five provider errors") from error
            sleep(poll_seconds)
            continue
        consecutive_errors = 0
        if pod is None:
            absent_reads += 1
            if absent_reads >= 2:
                log("absence_verified", pod_id=pod_id, consecutive_reads=absent_reads)
                return
            log("absence_pending_confirmation", pod_id=pod_id)
            sleep(poll_seconds)
            continue
        absent_reads = 0
        current = now()
        if attempted_at is not None and current - attempted_at >= 180:
            raise RuntimeError("termination requested but pod still listed after 180 seconds")
        if current >= deadline:
            if attempted_at is None:
                attempted_at = current
            log("termination_requested", pod_id=pod_id, name=name,
                provider_status=pod.get("desiredStatus"))
            try:
                command("pod", "delete", pod_id)
            except (subprocess.SubprocessError, OSError) as error:
                # Even an error can race a successful provider-side deletion.
                # Re-read the list rather than claiming either outcome.
                log("termination_command_failed", error_type=type(error).__name__)
        else:
            log("waiting", seconds_to_deadline=round(deadline-current, 1),
                provider_status=pod.get("desiredStatus"))
        sleep(min(poll_seconds, max(0.1, deadline-current)) if current < deadline
              else poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--deadline-utc", required=True, type=utc_deadline)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll interval must be between 1 and 60 seconds")
    if args.deadline_utc > time.time() + 86400:
        parser.error("deadline must be within the next 24 hours")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("x") as output:
        def log(event, **fields):
            record = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "event": event, **fields}
            output.write(json.dumps(record) + "\n")
            output.flush()
        log("started", pod_id=args.pod_id, name=args.pod_name,
            deadline_epoch=args.deadline_utc)
        try:
            watch(args.pod_id, args.pod_name, args.deadline_utc,
                  args.poll_seconds, log)
        except Exception as error:
            log("failed", error_type=type(error).__name__, message=str(error))
            raise


if __name__ == "__main__":
    main()
