# Independent work and spare GPU groups

The deployed configuration uses two independent instances of the keepalive program: physical devices 0–3 and physical devices 4–7. Each instance has its own root process, workers, log and guard lock. The full pipeline owns only the first group. The second group continues keepalive until explicitly borrowed.

Run an unrelated task from another terminal:

```bash
bash run_spare_task.sh -- /path/to/python your_job.py
# Or:
bash run_spare_task.sh -- bash your_job.sh
```

The command inherits `CUDA_VISIBLE_DEVICES=4,5,6,7`; inside PyTorch these appear as logical devices 0–3. Preserve that device selection. The wrapper returns the task's exit status. A second simultaneous borrower is rejected by the group lock.

Borrowing pauses only the spare keepalive tree. Normal completion, command failure and handled interruption clean up only that borrowed command and restore only the spare tree. A separate watchdog handles abrupt guard death. A borrowed task neither inherits the full run's cleanup tag nor causes its sidecar supervision to fail. Main-stage transitions and main completion also leave the borrowed task alone. If the main run finishes while a borrowed task is active, work-device keepalive resumes immediately and spare-device keepalive resumes when its borrower finishes. When both groups are idle, all eight devices run keepalive.

Pause/resume uses SIGSTOP/SIGCONT: the small keepalive allocations remain resident. GPU selection is cooperative process isolation, not a security boundary. Both tasks still share host CPU, memory and storage; exhausting those shared resources can affect either job. This feature does not survive the loss of the entire node/container without restarting the services.

Configuration is private deployment state in `runtime/site.env`:

```bash
export AOTD_KEEPALIVE_SCRIPT="$AOTD_ROOT/runtime/independent_keepalive/work.py"
export AOTD_SPARE_KEEPALIVE_SCRIPT="$AOTD_ROOT/runtime/independent_keepalive/spare.py"
export AOTD_GUARD_GPUS=0,1,2,3
export AOTD_INDEPENDENT_SPARE_KEEPALIVE=1
```

These paths must identify **separate, already-running** supervisors, restricted to their own physical devices. The deployment migrated the former shared supervisor into two instances of the same existing keepalive implementation. Setting the variables alone does not split a running supervisor. Scoped guards refuse roots that own devices outside their declared group. Keep source scripts, environment configuration and logs on persistent storage.

The full entrypoint defaults to four independent construction workers and four transfer/filter workers. Four evaluation models run concurrently, one per work device. Student training retains four-rank distributed training and runs Instruct then AoTD sequentially, preserving effective batch size 48 and checkpoint compatibility.

Raw validation summaries are stored in `reproduction/evidence/`. Runtime journal/model paths and infrastructure identities are intentionally not included in public evidence.
