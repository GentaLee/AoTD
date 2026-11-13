# 交付与复现索引 / Delivery and reproduction index

## 当前交付 / Current delivery

2026-11-13 核对：工程全量执行成功不等于严格论文复现成功。五源协议排除 ANetQA 和固定清单内 22 条异常 EgoQA 媒体。原作者完整处理后训练池、抽样 ID 与 few-shot 示例未公开；当前为工程重建。最新 [结果报告](FULL_V2_RESULTS_20261113.md) 与 [审计 JSON](../reproduction/evidence/full_v2_results.json) 优先于早期 smoke 快照。

The full engineering pipeline completed, but this is not an exact paper reproduction. Five-source reconstruction excludes ANetQA and 22 audited invalid EgoQA records. Author training data/sample IDs/few-shot examples remain unavailable. The latest results supersede historical smoke snapshots.

| 交付 / Deliverable | 位置 / Location | 范围 / Scope |
|---|---|---|
| 全量、制数、smoke 入口 / Entrypoints | `run_aotd_full.sh`, `run_aotd_build_data.sh`, `run_aotd_smoke.sh` | 已提交 / Tracked |
| 离线安装与资源 / Offline setup | `setup_offline.sh`, `setup_agent_offline.sh`, `requirements-runtime.txt`, `requirements-agent.txt`, [下载清单](../downloads/README.md) | 脚本、版本和清单；二进制不在 Git / Scripts and metadata, no binaries |
| 模型、数据来源 / Asset sources | [状态文档 §2、§7](REPRODUCTION_STATUS.md) | 教师、学生、五源公开入口 / Public model and dataset links |
| 并行与恢复 / Parallelism and recovery | [并行设计](PARALLEL_CONSTRUCTION.md), [独立借卡](SPARE_GPU_TASKS.md), [开发记录](DEVELOPMENT_LOG.md) | 四 worker、四 rank、checkpoint 与 guard / Workers, ranks, checkpoints, guards |
| 第二轮开发与验收 / V2 implementation and validation | [开发](CONSTRUCTION_V2.md), [验收](CONSTRUCTION_V2_VALIDATION.md) | 修复、局限、实测恢复 / Repairs, limitations, tested resume |
| 第一轮结果 / V1 results | [第一轮报告](FULL_RUN_RESULTS_20261110.md) | 历史完成结果 / Historical completed run |
| 第二轮结果与逐题对照 / V2 results and predictions | [结果](FULL_V2_RESULTS_20261113.md), [CSV](../reproduction/evidence/full_v2_predictions.csv) | 4,996 个 ID，标签、视频哈希及五列预测 / IDs, labels, media hashes, five prediction columns |
| 数据、轨迹、评测与日志 / Data, traces, evaluations and logs | [Release 与校验说明](ARTIFACT_DELIVERY_20261113.md) | 完整交付，私有信息脱敏 / Full delivery with private identifiers sanitized |
| 模型与训练 checkpoint / Weights and training checkpoints | 部署卷 / Deployment storage | 按要求不提交 / Excluded as requested |

## PR 依赖 / PR dependencies

[PR #5](https://github.com/GentaLee/AoTD/pull/5) 提交运行实现，目标为 `integration/offline-reproduction`；[PR #6](https://github.com/GentaLee/AoTD/pull/6) 在 #5 分支上补充第二轮结果和入口文档。#6 不是可以独立应用到原始主分支的完整代码补丁。合入实现后应将结果 PR 的 base 调整到对应集成分支并复核差异；本次不自动合并。

PR #6 is stacked on implementation PR #5, not a standalone implementation patch against the original main branch. After integrating #5, retarget/recheck #6 against the integration branch. No automatic merge is performed.

## 第二轮配置 / V2 configuration

先在持久化卷上准备 checkout、离线环境、模型、五源标注和媒体。版本见[状态文档](REPRODUCTION_STATUS.md)。首次部署不是仅 clone 后即可离线运行；缺少资源会明确停止。

Prepare the checkout, offline environment, models, annotations and media on persistent storage. A clone alone is not an offline asset bundle. Missing resources cause an explicit failure.

已有兼容资源可以用 `use_local_assets.sh` 关联，但它会链接整个 `data/`。**新提示词实验不能把可写制数目录链接到第一轮 `data/` 后直接运行**；必须隔离新 `data/construction/`、`data/five_source_sources/`、`data/five_source_train/` 与 `results/`，仅复用只读原始标注、媒体和模型。`--new-run` 仅创建新运行目录，不会自动隔离制数目录。

`use_local_assets.sh` links the entire data directory. For changed prompts, isolate construction journals, merged sources, training pools and results; reuse only immutable inputs/models. `--new-run` creates a run directory, not a fresh construction dataset. Incompatible fingerprints are rejected rather than silently mixed.

将以下参数放到不提交的 `runtime/site.env`，并按[独立借卡文档](SPARE_GPU_TASKS.md)配置两个已启动、分别限制设备的保活 supervisor：

Set these private runtime parameters and configure the two already-running device-scoped keepalive supervisors described in the spare-group guide:

```bash
export AOTD_REQUIRE_CONSTRUCTION_QUALITY_GATE=1
export AOTD_CONSTRUCTION_WORKERS=4
export AOTD_TEXT_WORKERS=4
export AOTD_GUARD_GPUS=0,1,2,3
export AOTD_INDEPENDENT_SPARE_KEEPALIVE=1
export AOTD_PROGRESS_INTERVAL_SECONDS=30
# Optional: validated completed QA-only control; omit to train Instruct anew.
# export AOTD_REUSE_INSTRUCT_DIR=/absolute/persistent/path/to/previous/instruct_s42
```

```bash
bash run_aotd_build_data.sh --check  # construction inputs, before building CoT
bash run_aotd_smoke.sh              # real short training/reload validation
bash run_aotd_full.sh               # construct if needed, train, evaluate; resume compatible work
# Once final training inputs exist:
bash run_aotd_full.sh --check
# Explicitly select an existing managed run:
bash run_aotd_full.sh --resume /absolute/persistent/path/to/results/full_RUN
# In another terminal, optional independent spare task:
bash run_spare_task.sh -- bash your_job.sh
```

主入口输出实际结果路径。使用 `tail -n 40 -f /absolute/persistent/path/to/results/full_RUN/run.log` 观察；Ctrl-C 只退出查看器。来源和相依阶段依次执行，阶段内部四路并行；Instruct/AoTD 默认依次训练，四模型评测并行。保活是资源管理，不计入有效训练吞吐。

The launcher prints its result directory. Follow its run.log with tail; Ctrl-C exits only the viewer. Sources/dependent stages are sequential, workers within stages are parallel, student runs are sequential, and model evaluations are concurrent. Keepalive is not useful training throughput.

## 产物边界 / Artifact availability

部署端保存最终模型 `aotd_s42/`、完整 `checkpoint-3400`/`checkpoint-3452`、训练完成标记、四份 `*_val_full.json`、`full_summary.json`、`run_state.json`、`run.log`、质量门控与对照复用检查。制数 journal 与训练池在 `data/`。第一轮结果保留。完整数据、轨迹、评测与日志通过 [Release](ARTIFACT_DELIVERY_20261113.md) 交付。模型与两个 checkpoint 按要求不提交；视频、环境与私有配置仍由使用者单独准备。

Deployment storage retains weights, complete checkpoints, completion markers, evaluations, summaries, logs, gate/reuse reports, construction journals and training data. First-run artifacts are preserved. Full data, traces, evaluations and logs are provided as release assets. Model weights and both training checkpoints are explicitly excluded. External media/models/environments still require separate provisioning; see the delivery guide for path remapping and verification.
