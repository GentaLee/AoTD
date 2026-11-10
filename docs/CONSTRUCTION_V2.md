# 制数接口修复与第二轮实验 / Construction contract repair and second experiment

## 中文

第一轮完整训练评测成功，但 AGQA 25,000 条只产生 380 条合格 CoT。审计发现工具描述与返回类型不一致、开放问答布尔答案漏判，以及大量模型回复说明文字而非程序。此轮不覆盖第一轮 journal、训练池、模型或评测。

### 改动

- 程序提示词准确说明 `Query_Actions`/`Query_Objs` 返回文本，`VideoClip` 不可迭代或按下标访问；对象存在不等于触摸或时间顺序。列出允许的操作，保持原执行白名单，不为提高产量取消限制。
- 对 DeepSeek 的 assistant 输入追加固定函数签名，生成后拼回同一签名，让模型完成函数体；仍通过原 AST 检查。第一版仅改提示词的验证失败，失败记录保留，未用于全量。
- 仅对 AGQA 无选项且标签为 yes/no 的题，将完整 `True/False` 输出对应到 yes/no；不做子串匹配、不把任意解释或对象答案视为正确。
- 修复 `Video_summary(clip, query)` 的 query 被忽略问题：按问题请求真实可见的对象、动作和先后顺序；无 query 时保留原泛化摘要。程序先收集这类证据再回答，避免只有一个答案词却要求转换器编写解释。
- 转换提示词要求只使用轨迹中的观察，不再强制生成不存在的坐标，也不允许把程序输出改为臆测的“人类答案”。后续抽查发现原筛选器只看问题和解释，不能核验轨迹与答案一致性；本轮改为同时提供真实轨迹和程序答案，要求证据充分、结论一致、时序与否定正确，仍只接受严格二元 True。标准答案不传给筛选器。
- 选择题转换及筛选上下文显式带入全部选项和程序实际选中的文本，避免把 `C` 猜成某个词。解析依据 program_output，不读取标准答案；有标准答案变化不影响上下文的回归测试。
- 新入口可启用 `AOTD_REQUIRE_CONSTRUCTION_QUALITY_GATE=1`：训练前核验三源构建覆盖、正确轨迹的后续阶段覆盖、无运行错误、导出 ID 一致，以及各来源合格 CoT 不低于旧基线（2,205 / 1,814 / 380）。这是完整性及产量回退检查，不是自动语义质量认证。
- 可用 `AOTD_REUSE_INSTRUCT_DIR` 指定完整 QA-only 对照。只在 QA 内容/顺序/实际媒体路径、Base、训练配置、训练实现和导出完成检查一致时复用；任何不一致都会停止，不会把旧权重冒充本轮新训练。AoTD 仍从 Base 初始化。

### 并行、恢复和资源归属

构建、转换、筛选四 worker；评测按模型并行；训练四 rank、有效 batch 48。逐条 fsync、来源指纹检查、完整训练 checkpoint、评测 journal 均保留。提示词变更必须使用新制数目录，不加入允许复用旧轨迹的迁移白名单。

新部署与旧部署共用保活 guard 锁，避免两个入口同时借用同一工作组。0–3 工作组与 4–7 借用组独立；中断后恢复所拥有的组，不打断另一组任务。工作组与借用组均空闲时八卡保活。共享 CPU/RAM/磁盘与容器级退出的限制不变。

### 验收与启动顺序

固定开发验证子集为 AGQA 256 条、NExT-QA 64 条、STAR 64 条，按 sample ID 的固定哈希选择，与第一轮相同 ID 逐条对照。这些样本来自训练划分，并用于提示词迭代，因此不是无偏泛化评估。验证 journal 与正式训练池隔离，不混入正式产物。完成程序构建、转换、筛选、真实混合记录两步训练、完整 checkpoint 保存及重载评测后才启动全量。

本轮还实际执行构建中的 SIGTERM，核验旧 journal 字节前缀、八个原保活 worker 恢复，再从原目录续接。最终数值与运行观察以随 PR 的验收证据为准；验证集增益不保证全量达到论文 CoT 数量或学生准确率。

## English

The first full training/evaluation run completed, but AGQA yielded only 380 accepted rationales from 25,000 questions. The audit found mismatched tool return-type descriptions, missed boolean equivalents in open-answer matching, and frequent prose instead of executable programs. This experiment preserves all first-run data, models and results.

Question-conditioned Video_summary now actually forwards the question to the visual model, while preserving the no-query fallback. Programs collect relevant visual evidence before answering rather than asking transfer to invent reasoning from a bare answer. The revised prompt documents real text/VideoClip contracts and the existing operation allowlist. A fixed assistant function-signature prefix constrains DeepSeek to completing a function body; AST validation remains mandatory. AGQA yes/no labels accept only complete boolean equivalents, not arbitrary prose. Trace transfer must stay within recorded observations and must not invent coordinates or override the recorded answer. The original judge received only the question and rationale. Inspection found accepted temporal reversals and explicit unsupported-answer explanations, so the judge now also receives the trace and program answer and requires evidential support and answer/temporal/negation consistency. It still accepts only a strict True verdict and never receives the ground-truth label. Multiple-choice transfer/filter inputs now include all choices and the program-selected answer text, resolved from program_output without consulting the ground-truth label.

The optional pre-training gate checks complete construction, downstream coverage of correct traces, operational errors, exported IDs and per-source baseline yield floors. It does not certify semantic quality. Optional Instruct reuse validates QA content/order/resolved media paths, Base identity, recipe, training implementation and completed exports; AoTD still starts from Base.

Four-worker construction/text stages, concurrent model evaluation, four-rank training, durable journals and checkpoint recovery are preserved. Changed prompts require a fresh construction directory. Old and new deployments share guard locks; work and spare groups remain independent.

A fixed 384-question real-data development pilot (256 AGQA, 64 NExT-QA, 64 STAR) is compared with matching first-run records and kept separate from production journals. It is sampled from training data and used for prompt iteration, so it is not an unbiased generalization evaluation. Validation includes controlled SIGTERM/resume with preserved journal bytes and restored keepalive, text stages, two-step mixed-data training, checkpoint saving and reloaded evaluation. Full execution follows successful validation; pilot gains do not guarantee full-data or paper-level results.


## 可复现验证命令 / Reproducible validation commands

在已配置的离线 checkout 中，`PILOT_QA` 指向与正式 journal 隔离的固定开发样本，`PILOT_OUT` 为新验证目录。所有 GPU 命令都应在项目的 scoped keepalive guard 内运行。

```bash
export AOTD_CONSTRUCTION_WORKERS=4 AOTD_TEXT_WORKERS=4
for stage in construct transfer filter export; do
    env/bin/python -B tools/build_cot.py --stage "$stage" \
        --qa "$PILOT_QA" --output-dir "$PILOT_OUT"
done
```

CPU 回归在目标运行节点隐藏 CUDA，并将 driver 的默认 worker 数设为 1；并行测试自身显式指定所测 worker 数，避免把真实部署默认值混入 mock 用例。

```bash
AOTD_CONSTRUCTION_WORKERS=1 AOTD_TEXT_WORKERS=1 \
AOTD_OWLV2_MODEL_DIR="$PWD/models/teachers/OWLv2" CUDA_VISIBLE_DEVICES= \
    env/bin/python -B -m unittest discover -s tools -p 'test_*.py'
```

Full continues to use four workers. These CPU-only test defaults do not change deployment concurrency. The pilot is sampled by the lowest SHA256 of `contract-v2-seed42:<sample_id>` within each source, using 256/64/64 records. Ground-truth labels are used for post-execution matching, never passed to the program planner or visual observation prompts.


## 验证迭代边界 / Validation iteration boundaries

- 首轮只改接口提示词：40 条均因输出格式等拒绝，未进入全量。
- 固定函数前缀后：384 条中 86 条答对，但只生成 3 条合格 CoT；拒绝用这个低产量版本直接训练全量。
- 补充真实问题相关视觉描述后：384 条中 113 条答对，初次筛选 75 条。抽查又发现选择题只有字母、缺少选项文本的上下文问题，随后仅重放转换/筛选；构建轨迹保留并记录输入哈希。
- 所有候选的试验记录保留。最终候选统计、checkpoint 重载及全量恢复观察单独保存到验收证据，不把中间候选当作正式结果。

The first prompt-only attempt rejected all 40 processed records. Adding the function prefix produced 86 correct answers on 384 questions but only three accepted rationales, so it was not promoted. Question-conditioned visual evidence produced 113 correct answers and initially 75 accepted rationales. Inspection then found missing choice text in the text stages; those stages were replayed with explicit program-selected choice context and preserved construction-input hashes. Intermediate candidate outputs are not production results.

轨迹一致性抽查不等于逐帧人工核实。视觉教师仍可能产生错误描述，二元筛选也不是事实正确性证明。记录完整轨迹、标准答案门控和不低于基线的产量检查降低工程失误风险，但不能保证达到作者模型表现。

Trace-consistency review is not frame-by-frame human verification. Visual teachers can still hallucinate, and the binary judge is not a factuality certificate. Durable traces, answer matching and yield floors do not guarantee author-level model performance.

增加筛选上下文的回归测试后，最终 CPU/GPU 验收和第二轮全量状态见随 PR 的验证证据。

Final CPU/GPU validation, including the added judge-context regression, and second-run observations are recorded separately in the PR evidence.


### 筛选协议的明确变更 / Explicit judge protocol change

带选项文本的候选产生 83 条接受记录，但人工轨迹抽查发现其中存在反转 before/after、以及明确承认答案缺少支持的解释。这不是可接受的产量提升，因此未晋级全量。最终候选保留构建和转换结果，仅重放更严格的轨迹一致性筛选，并记录输入哈希。此变更与第一轮筛选协议不同，必须在结果比较中披露，不能声称完全按作者原始 CoT 复现。

The choice-aware candidate accepted 83 records, but trace review found a before/after reversal and an explanation explicitly stating that its answer lacked support. That candidate was not promoted. The final candidate replays only filtering with explicit trace-consistency requirements and records the unchanged input hashes. This differs from the first-run judge protocol and must be disclosed in comparisons; it is not an exact reproduction of the authors' original CoT corpus.
