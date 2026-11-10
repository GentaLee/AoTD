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
