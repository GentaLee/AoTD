# 第二轮开发验证结果 / Second experiment development validation

本文记录第二轮的开发验证，不是第二轮全量成绩。实现和原因见 [开发说明](CONSTRUCTION_V2.md)，第一轮完整评测仍见 [历史结果](FULL_RUN_RESULTS_20261110.md)。

This report covers development validation, not second-run full results. The first experiment's complete metrics remain available in the historical report.

## 固定真实样本对照 / Matched real-sample comparison

按固定 sample ID 哈希选取训练划分中的 384 题，逐一与旧版相同 ID 对照。该子集参与提示词迭代，不能当作独立测试集。

| 来源 / Source | 题数 / Questions | 旧正确答案 / Old correct | 新正确答案 / New correct | 旧 CoT / Old CoT | 新 CoT / New CoT |
|---|---:|---:|---:|---:|---:|
| AGQA | 256 | 9 | 52 | 3 | 35 |
| NExT-QA | 64 | 15 | 36 | 7 | 33 |
| STAR | 64 | 12 | 25 | 3 | 15 |
| 合计 / Total | 384 | 36 | 113 | 13 | 83 |

构建 326 条可执行、58 条程序拒绝、0 条运行错误；113 条正确轨迹全部完成转换和筛选。新产量约为旧版 6.38 倍，但这一比例不能外推到全量。候选的拒绝与修改过程详见开发说明；最终筛选同时提供程序答案、选项及执行轨迹，协议与第一轮不同。

The matched development pilot yields 83 versus 13 accepted rationales (6.38×), with 326 executed programs, 58 program rejections and no operational errors. All 113 correct traces completed transfer and filtering. This ratio is not a full-data forecast. The final judge now receives the trace and proposed answer, so the filtering protocol differs from the first experiment.

## 质量边界 / Quality limits

明确发现的两条错误（反转 before/after、明确承认答案无证据）在新版筛选中均被拒绝。另抽查九条接受记录，仍看到视觉教师内部矛盾、时间前提缺失和解释推断过度。模型二元筛选不是事实质量认证，不能把 83 条称为经过人工逐帧核验的正确解释，也不能宣称学生效果已经提升。

The two identified defects (a before/after reversal and an explicitly unsupported answer) are rejected by the revised judge. Review of nine further accepted records still found contradictory teacher observations, omitted temporal prerequisites and over-inference. Binary model judgments do not certify factual correctness; these are not 83 frame-by-frame human-verified rationales. No student improvement is claimed before full evaluation.

## 运行验证 / Execution validation

最终 CPU 回归 88/88 通过，耗时 12.037 秒。四卡混合 QA/CoT 两步训练、完整 checkpoint、最终模型重载以及生产启动/恢复的原始汇总由 [验收 JSON](../reproduction/evidence/construction_v2_validation.json) 和 [全量启动观察](../reproduction/evidence/full_v2_startup.json) 记录；只将实际通过的状态写入证据。

CPU regressions passed 88/88 in 12.037 seconds. The linked evidence records actual four-rank mixed QA/CoT training, checkpoint/export validation, reloaded evaluation and production startup/resume observations. Smoke evaluation checks execution only; four questions do not estimate model quality.

第二轮将重新构建三源 CoT，使用五源全部 133,654 条有效 QA。Instruct 仅在 QA 内容/顺序/媒体和 Base/训练协议/实现一致时复用旧完整对照；AoTD 从 Base 独立训练一轮，再做四模型完整 val 评测。训练前的产量门槛只防止明显回退，不能保证论文参考量或效果。

The second experiment rebuilds CoT for three sources while keeping all 133,654 valid QA records from five sources. A completed Instruct control is reused only after QA/base/recipe/implementation equality checks. AoTD trains independently from Base for one epoch, followed by all four full validation evaluations. Yield floors prevent obvious regression; they do not guarantee paper-level volume or accuracy.


四卡两步训练的 loss 为 15.7148、15.6562（均有限），有效 batch 48、BF16、ZeRO-2。训练运行计时 100.2645 秒，完整 step-2 checkpoint 和最终导出均提交；导出模型实际重载，4 道 smoke 题全部完成、无无效输出。八卡保活在 smoke 结束后恢复。全量生产已经启动，并在 65 条已提交记录后执行 SIGTERM 验收：退出 143、八个原保活 worker 恢复，150,979 字节旧 journal 的 SHA256 不变；随后同一入口恢复同一运行目录，详细持续观察见链接的启动证据。

Four-rank, two-step training produced finite losses of 15.7148 and 15.6562 with effective batch 48, BF16 and ZeRO-2; recorded training runtime was 100.2645 seconds. A complete step-2 checkpoint and final export were committed, and the exported model reloaded and completed all four smoke questions with no invalid outputs. All eight keepalive workers resumed afterward. The production full run was then started and interrupted with SIGTERM after 65 committed records: exit 143, all original eight keepalive workers restored, and the 150,979-byte journal prefix unchanged. The same entrypoint resumed the same run directory; sustained observations are recorded in the linked startup evidence.

恢复后连续观察 180.08 秒，记录从 65 增至 225（新增 160），无重复 ID 或运行错误，后四卡保活身份及运行状态保持不变。此时全量仍在构建阶段，尚未完成训练评测。

After progress resumed, 180.08 seconds of observation showed 65→225 records (160 new), zero duplicate IDs or operational errors, and unchanged running spare keepalive workers. Full construction remains in progress; full training/evaluation is not yet complete.
