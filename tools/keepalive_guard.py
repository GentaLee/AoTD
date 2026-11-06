#!/usr/bin/env python3
"""Pause a known keep-alive process tree while running one exclusive GPU command.

Uses SIGSTOP/SIGCONT, so keep-alive allocations remain in GPU memory. A detached
watchdog resumes only processes paused by this invocation if the guard exits
unexpectedly. Requires Linux /proc and nvidia-smi; uses no third-party modules.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True


def note(message):
    print(f"[keepalive-guard] {message}", flush=True)


def proc(pid):
    try:
        # comm may itself contain spaces and parentheses.
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                    pgrp=int(fields[2]), session=int(fields[3]), start=int(fields[19]),
                    uid=Path(f"/proc/{pid}").stat().st_uid)
    except (OSError, ValueError, IndexError):
        return None


def processes():
    return {p["pid"]: p for name in os.listdir("/proc")
            if name.isdecimal() and (p := proc(int(name))) is not None}


def same_process(identity):
    current = proc(identity["pid"])
    return current is not None and current["start"] == identity["start"]


def tree(root, table):
    result = {root}
    while True:
        expanded = result | {pid for pid, p in table.items() if p["ppid"] in result}
        if expanded == result:
            return result
        result = expanded


def keepalive_roots(script):
    roots = []
    expected = str(Path(script).resolve(strict=True))
    for pid, p in processes().items():
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            argv = [a.decode(errors="replace") for a in argv if a]
            if not argv or not Path(argv[0]).name.startswith("python"):
                continue
            if any(a in argv for a in ("--monitor", "--plan-only", "--self-test")):
                continue
            # Match the interpreter's script argument, not a flag value passed
            # to this guard or another Python program.
            position = 1
            while position < len(argv) and argv[position].startswith("-"):
                option = argv[position]
                if option in ("-c", "-m"):
                    break
                position += 2 if option in ("-W", "-X") else 1
            candidate = argv[position] if position < len(argv) else ""
            if candidate.startswith("/") and str(Path(candidate).resolve()) == expected:
                if p["state"] not in ("Z", "X"):
                    roots.append(p)
        except (OSError, ValueError):
            continue
    return roots


def gpu_pids():
    run = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=15,
    )
    values = [line.strip() for line in run.stdout.splitlines() if line.strip()]
    if any(not value.isdecimal() for value in values):
        raise RuntimeError(f"Cannot interpret nvidia-smi process list: {values}")
    return {int(value) for value in values}


def gpu_status():
    subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
         "--format=csv"], check=True, timeout=15,
    )


def resume(paused, log=note):
    failures = []
    for identity in reversed(paused):
        if same_process(identity):
            try:
                os.kill(identity["pid"], signal.SIGCONT)
                log(f"resumed keep-alive PID {identity['pid']}")
            except ProcessLookupError:
                pass
            except OSError as exc:
                failures.append(f"{identity['pid']}: {exc}")
    if failures:
        raise RuntimeError("Could not resume keep-alive: " + "; ".join(failures))


def command_members(identity):
    if not identity:
        return []
    # Torch elastic launches each rank with start_new_session=True. The inherited
    # environment tag also finds these ranks after their launcher has exited and
    # they have been reparented. Never print the contents of /proc/*/environ.
    tag = ("AOTD_GUARD_RUN_ID=" + identity["run_id"]).encode() if identity.get("run_id") else None
    members = []
    for p in processes().values():
        if (p["start"] < identity["start"] or p["uid"] != identity.get("uid", os.getuid())
                or p["state"] in ("Z", "X")):
            continue
        matched = p["session"] == identity["pid"]
        if not matched and tag:
            try:
                matched = tag in Path(f"/proc/{p['pid']}/environ").read_bytes().split(b"\0")
            except OSError:
                pass
        if matched:
            members.append(p)
    return members


def clean_command(identity, log=note):
    """Stop the command session, including workers left after its leader exits."""
    members = command_members(identity)
    if not members:
        return
    log("stopping command session PIDs " + ",".join(str(p["pid"]) for p in members))
    for sig, timeout in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
        for p in command_members(identity):
            try:
                os.kill(p["pid"], sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        while command_members(identity) and time.monotonic() < deadline:
            time.sleep(0.1)
        if not command_members(identity):
            return
    raise RuntimeError("Command workers remain alive; inspect the recorded command session")


def watchdog(read_fd, log_path):
    """EOF means the guard died; updates are written before any process is stopped."""
    os.setsid()
    state = {"paused": [], "command": None}
    done = False
    with os.fdopen(read_fd, "r") as pipe:
        for line in pipe:
            state = json.loads(line)
            if state.get("done"):
                done = True
                break
    if not done:
        with open(log_path, "a", buffering=1) as stream:
            def log(message):
                stream.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {message}\n")
            log("Guard disappeared; recovering command and keep-alive")
            try:
                clean_command(state.get("command"), log)
            except Exception as exc:
                log(f"Recovery blocked; keep-alive remains paused because command cleanup failed: {exc}")
                raise
            resume(state.get("paused", []), log)
            log("Recovery complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keepalive-script", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    state_dir = Path(args.state_dir).resolve()
    if not state_dir.is_relative_to("/volume"):
        parser.error("--state-dir must be beneath a mounted /volume path")
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = open(state_dir / "keepalive-guard.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another GPU command holds the keep-alive guard lock")

    roots = keepalive_roots(args.keepalive_script)
    if len(roots) > 1:
        raise RuntimeError("Multiple keep-alive roots found; refusing an ambiguous pause")
    root = roots[0] if roots else None
    allowed = tree(root["pid"], processes()) if root else set()
    other = gpu_pids() - allowed
    if other:
        raise RuntimeError(f"Other GPU processes are running; refusing to proceed: {sorted(other)}")
    note(f"keep-alive root: {root['pid'] if root else 'not running'}; GPU memory stays allocated during pause")
    gpu_status()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = state_dir / f"guard-{stamp}-{os.getpid()}.json"
    recovery_log = state_dir / f"recovery-{stamp}-{os.getpid()}.log"
    state = {"guard_pid": os.getpid(), "run_id": uuid.uuid4().hex, "paused": [], "command": None,
             "keepalive_script": str(Path(args.keepalive_script).resolve()), "status": "preparing"}
    read_fd, write_fd = os.pipe()
    watchdog_pid = os.fork()
    if watchdog_pid == 0:
        os.close(write_fd)
        # Keep the inherited lock until emergency recovery is complete.
        try:
            watchdog(read_fd, recovery_log)
        finally:
            os._exit(0)
    os.close(read_fd)
    pipe = os.fdopen(write_fd, "w", buffering=1)

    def update():
        record.write_text(json.dumps(state, indent=2) + "\n")
        pipe.write(json.dumps(state) + "\n")
        pipe.flush()

    interrupted = []
    def request_stop(signum, _frame):
        interrupted.append(signum)
    previous_handlers = {s: signal.signal(s, request_stop)
                         for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    command_pid = None
    exit_code = 1
    try:
        update()
        if root:
            # Stop the root before taking descendants, preventing further spawns.
            for _ in range(5):
                if not same_process(root):
                    raise RuntimeError("Keep-alive root changed during the pause")
                table = processes()
                members = tree(root["pid"], table)
                ordered = [root["pid"]] + sorted(members - {root["pid"]})
                added = False
                for pid in ordered:
                    p = table.get(pid)
                    if p is None or p["state"] in ("Z", "X", "T", "t"):
                        continue
                    if any(p["pid"] == old["pid"] and p["start"] == old["start"]
                           for old in state["paused"]):
                        continue
                    state["paused"].append(p)
                    update()  # Persist and notify the watchdog before SIGSTOP.
                    if same_process(p):
                        try:
                            os.kill(pid, signal.SIGSTOP)
                            note(f"paused keep-alive PID {pid}")
                        except ProcessLookupError:
                            pass
                    added = True
                if not added:
                    break
                time.sleep(0.1)
            deadline = time.monotonic() + 5
            while True:
                active = [p for p in state["paused"] if same_process(p)
                          and proc(p["pid"])["state"] not in ("T", "t", "Z", "X")]
                if not active:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Keep-alive processes did not stop")
                time.sleep(0.1)
        if interrupted:
            exit_code = 128 + interrupted[0]
            return exit_code
        if root:
            allowed |= tree(root["pid"], processes())
        if gpu_pids() - allowed:
            raise RuntimeError("Another GPU process appeared during the pause")
        state["status"] = "paused"
        update()
        note("keep-alive paused; starting the requested command")
        # A gate prevents exec until the watchdog knows the command PID/session.
        gate_read, gate_write = os.pipe()
        command_pid = os.fork()
        if command_pid == 0:
            try:
                os.close(gate_write)
                os.close(pipe.fileno())
                lock.close()
                os.setsid()
                for sig in previous_handlers:
                    signal.signal(sig, signal.SIG_DFL)
                go = os.read(gate_read, 1)
                os.close(gate_read)
                if go != b"G":
                    os._exit(125)
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                           AOTD_GUARD_RUN_ID=state["run_id"])
                os.execvpe(command[0], command, env)
            except BaseException as exc:
                print(f"[keepalive-guard] command exec failed: {exc}", file=sys.stderr, flush=True)
                os._exit(127)
        os.close(gate_read)
        state["command"] = proc(command_pid)
        if state["command"] is None:
            raise RuntimeError("Command exited before its execution gate opened")
        state["command"]["run_id"] = state["run_id"]
        state["status"] = "running"
        update()
        os.write(gate_write, b"G")
        os.close(gate_write)
        while True:
            if interrupted:
                note(f"received signal {interrupted[0]}; stopping command and resuming keep-alive")
                clean_command(state["command"])
            pid, status = os.waitpid(command_pid, os.WNOHANG)
            if pid:
                result = os.waitstatus_to_exitcode(status)
                exit_code = (128 - result) if result < 0 else result
                if interrupted:
                    exit_code = 128 + interrupted[0]
                break
            time.sleep(0.2)
    finally:
        # Do not resume the load while an identified payload worker survives.
        # If this raises, EOF lets the watchdog retry cleanup before resuming.
        clean_command(state.get("command"))
        resume(state["paused"])
        state["status"] = "restored"
        state["exit_code"] = exit_code
        state["done"] = True
        update()
        pipe.close()
        os.waitpid(watchdog_pid, 0)
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
        note(f"keep-alive restored; state: {record}")
        gpu_status()
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        note(f"ERROR: {exc}")
        sys.exit(1)
