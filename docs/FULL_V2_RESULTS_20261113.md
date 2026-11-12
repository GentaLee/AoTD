# 第二轮完整结果 / Second full-run results

完成时间：2026-11-13 00:06:18 UTC+08:00（2026-11-12 16:06:18 UTC）。全流程 complete、exit 0、PASS，日志确认工作组保活恢复。第一轮也完成了完整训练评测；其问题是产量低、效果提升很小，不是最终运行失败。

Completed with exit 0 and PASS. Work-group keepalive restoration is recorded in the completion log. The first run also completed; its shortcomings were low rationale yield and weak effectiveness, not a failed final execution.

| 指标 / Metric | V1 | V2 |
|---|---:|---:|
| QA | 133,654 | 133,654 |
| NExT-QA CoT | 2,205 | 16,515 |
| STAR CoT | 1,814 | 12,256 |
| AGQA CoT | 380 | 3,271 |
| Total CoT | 4,399 | 32,042 |
| AoTD updates | 2,877 | 3,452 |
| AoTD training hours | 8.76 | 10.74 |
| Base correct / 4,996 | 3,121 | 3,121 |
| Instruct correct / 4,996 | 3,527 | 3,527 |
| AoTD correct / 4,996 | 3,529 | 3,547 |
| AoTD accuracy | 70.64% | 71.00% |
| Author accuracy | 77.62% | 77.62% |

CoT 数量增加到 7.28 倍。V2 比 V1 多答对 18 题（+0.3603 个百分点），比 Instruct 多 20 题（+0.4003 个百分点），仍比作者模型少 331 题（-6.6253 个百分点）。相对 Instruct，逐题独胜/独负为 271/251；相对 V1 为 264/246。不能将净提升解释成所有题目一致改善。

CoT yield increased 7.28×. V2 gains 18 correct answers over V1 (+0.3603 percentage points) and 20 over Instruct (+0.4003 points), but trails the author checkpoint by 331 answers (-6.6253 points). Paired wins/losses are 271/251 against Instruct and 264/246 against V1. These small net improvements do not establish robust or statistically significant benefits; no multi-seed experiment was performed.

| 类别 / Category | Instruct | V2 AoTD | 净增正确 / Net correct |
|---|---:|---:|---:|
| C (2,607) | 1,856 | 1,858 | +2 |
| T (1,612) | 1,072 | 1,082 | +10 |
| D (777) | 599 | 607 | +8 |

工程修复有效：接口、观察上下文与筛选修复后产量显著增加，全量恢复及并行策略运行完成。效果只显示单次小幅正收益，不能宣称严格论文复现成功。提示词、筛选规则、CoT 数量/长度和训练步数同时改变，无法把收益单独归因于某项修复。Instruct 是经一致性检查复用的同一对照，不是独立重复训练。

Engineering repairs improved yield and completed the parallel/resumable pipeline. Model effectiveness shows only a small single-run gain. Prompts, filtering, rationale volume/length and training steps changed together, so causal attribution to one repair is not possible. Instruct is a validated reused control, not an independent repeat. Five-source reconstruction excludes ANetQA and 22 audited invalid EgoQA media; author data and prompt examples remain unavailable. No strict paper reproduction claim is made.

## 审计与产物 / Audit and artifacts

四组各 4,996 个唯一 ID；逐题重算正确数、无效输出及 C/T/D 分类，与汇总一致；跨模型及 V1 的题目 ID、标签、视频哈希一致。训练 loss 有限，完成标记、最终 checkpoint 标记和导出文件大小通过检查。最终模型已实际加载完成全部评测；本次未重新对大型权重分片做全量哈希。保活结论来自完成日志，不代表未来任意时刻的资源状态。

Audit verified unique full coverage, recomputed totals/categories/invalid outputs and matching labels/video hashes across runs. Finite loss, completion markers, final checkpoint marker and export sizes passed. The final model actually completed full evaluation; large weight shards were not fully rehashed in this audit. Keepalive evidence describes completion time.

- [机器可读结果 / Machine-readable audit](../reproduction/evidence/full_v2_results.json)
- [逐题预测对照 / Per-question predictions](../reproduction/evidence/full_v2_predictions.csv): all 4,996 IDs, labels, media hashes and five model/run predictions; no private deployment paths.
- [开发与变更 / Development changes](CONSTRUCTION_V2.md)
- [Smoke 与恢复验收 / Smoke and resume](CONSTRUCTION_V2_VALIDATION.md)

部署侧保留 `results/<run>/aotd_s42/` 最终模型、checkpoint-3400/3452、四份 `*_val_full.json`、`full_summary.json`、`construction_quality_gate.json`、`instruct_reuse.json` 和 run.log。完整制数 journal 与筛选产物保留在 `data/construction/<source>/`，训练池在 `data/five_source_train/`。旧产物未覆盖。模型、视频、完整教师轨迹和私有配置不上传 Git；仓库保存审计摘要和可复核预测对照。

Deployment artifacts retain final weights, the last two full checkpoints, evaluation outputs, integrity/reuse reports and logs; construction journals and training data remain on persistent storage. Old artifacts are preserved. Git contains sanitized audit data and prediction comparisons, not large weights, videos, full teacher traces or private configuration.
