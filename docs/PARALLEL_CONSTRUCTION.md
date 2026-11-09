# Parallel construction and resume

The full entrypoint defaults to `AOTD_CONSTRUCTION_WORKERS=4`. Each spawned process sees exactly one of the four work devices and owns a complete teacher stack. Four independent samples can advance simultaneously. Student training remains four-device distributed training; transfer, filtering and evaluation retain their existing execution paths.

This mode requires enough memory on **each** work device for all construction teachers plus inference activations. It is not tensor parallelism. `AOTD_CONSTRUCTION_WORKERS=1 bash run_aotd_full.sh` retains the former serial, model-distributed path for smaller devices. The default is configured by the full entrypoint, not by generic imports or test fixtures.

Workers have separate Agent configurations and UniVTG scratch directories. Generated programs run on each process's main thread, preserving the existing SIGALRM timeout and isolating global trace state. CUDA is initialized only after spawn and device assignment. The coordinator bounds outstanding work to one sample per worker, dispatches another when a worker finishes, and alone writes the shared journal with flush/fsync. There is no concurrent JSONL writing and no dataset-sized queue of video tensors.

Completed records remain keyed by sample ID. Completion order can differ from input order. After interruption, durable successes and rejections are skipped; at most four uncommitted in-flight samples need repeating. A worker crash, broken pipe, model error or watchdog expiry fails the stage instead of declaring a partial pool complete. The surrounding guard cleans descendants and restores the original keepalive workload. Host/container failure remains outside an in-container watchdog's recovery capability.

The reviewed manifest migration permits only the exact prior and new driver/tool fingerprints, with unchanged models, prompts, inputs and protocol. Prior manifests are archived. It preserves old journal bytes; it does not claim bitwise identical floating-point inference across different placements. All teachers retain greedy decoding and the existing output validation.

Progress remains at the configured interval (30 seconds by default). The full command resumes the latest managed run. Do not use `--new-run` to resume this work. Four spare devices retain their supervised keepalive tasks during construction/training, and the guard restores all original keepalive workers after completion or failure.

Validation commands (on the target with its offline environment):

```bash
source runtime_env.sh
AOTD_OWLV2_MODEL_DIR="$PWD/models/teachers/OWLv2" CUDA_VISIBLE_DEVICES='' \
  env/bin/python -B -m unittest discover -s tools -p 'test_*.py'
```

The spawned scheduler tests cover out-of-order completion, unique results, early close and durable resume, worker death, operational-error propagation and the no-work fast path. Real-model throughput and restart measurements must be reported separately from these CPU regressions.
