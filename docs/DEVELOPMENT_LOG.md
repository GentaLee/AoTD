# 2026-11-11：修复制数契约并准备第二轮全量 / Repair construction and prepare a second full run

AGQA 低产量审计定位到工具返回类型描述、布尔答案匹配、程序生成格式和视觉摘要忽略问题等缺陷。新版本同时修复选择题转换缺少选项文本，以及筛选器缺少轨迹/程序答案的上下文。完整设计、迭代失败记录与边界见 [第二轮开发说明](CONSTRUCTION_V2.md)。第一轮产物全部保留。

固定开发子集 384 条：正确程序答案从 36 增至 113，最终 CoT 从 13 增至 83（AGQA 3→35，NExT-QA 7→33，STAR 3→15）。该子集参与了提示词迭代，不是独立测试集；仍发现教师描述与语义筛选的残余误判，不把数量提升当作最终效果证明。

保留四 worker 制数/转换/筛选、四 rank 训练、四模型并行评测、逐条 journal 和完整训练 checkpoint。新旧部署共用 guard 锁；工作组与借用组的资源所有权仍独立。第二轮重建三源 CoT，五源 133,654 条 QA 不变；经内容、顺序、媒体、Base、训练配方和实现校验后复用第一轮 Instruct 对照，AoTD 从 Base 重新训练。生产状态与中断/恢复观察将在启动后追加。

The second experiment repairs tool contracts, boolean matching, function completion, question-conditioned visual evidence, choice-aware transfer and trace-aware judging. Four-way execution and scoped keepalive recovery remain enabled. The development pilot improves from 13 to 83 accepted rationales but was used for prompt iteration and still contains teacher/judge errors. All first-run artifacts remain intact; only a verified identical QA-only control may be reused, while AoTD starts again from Base. See the linked development note for limitations and the validation evidence for measured checks.

---

# 2026-11-10：五源完整运行结束

两轮训练和四组完整 val 评测已完成，退出码 0，汇总 PASS，工作组保活已恢复。开发实现与验证见[独立开发报告](FULL_RUN_DEVELOPMENT_20261110.md)，最终指标与限制见[独立结果报告](FULL_RUN_RESULTS_20261110.md)。AoTD 70.64%，Instruct 70.60%，作者权重 77.62%；流程成功不等于论文指标复现成功。以下内容为按时间保留的历史记录，其中“尚未完成”指对应记录时点。

---

# 2026-11-09：后续阶段并行与独立借卡

审计发现转换、筛选均为单卡逐条处理，四组评测依次执行；训练已经是四卡分布式，两个实验依次使用完整四卡。现将转换与筛选接入同一个有界、多进程、单写者的持久化调度器；四组评测每卡一个模型并行，仍各自保留逐题 journal。训练协议、有效 batch 48、两个训练目标与 checkpoint 逻辑不变。

真实对照：64 条转换结果逐条一致，warm 时间 218.38 → 73.93 秒（2.95 倍）；1,024 条筛选判定全部一致，60.59 → 22.79 秒（2.66 倍）。小样本的总耗时受加载影响，原始数据同时记录 wall/warm 时间。转换、筛选分别从 32 条真实持久记录恢复，只新增剩下的 32 条，旧字节前缀保留，结果与对照一致。两份已存在的评测模型各跑 8 条真实问题，并验证重复启动跳过完成结果；学生模型尚未训练，因此不把这项 smoke 写成四份最终模型的完整评测。

将共享保活 supervisor 拆成两个独立组。`keepalive_guard --gpus` 只验证、暂停、恢复对应组，拒绝跨组 supervisor；工作组和借用组使用不同锁与清理标签。`run_spare_task.sh` 提供 4–7 卡借用入口。实际借卡任务正常完成、错误码 17 退出，以及借卡 guard 被 SIGKILL 后，后四卡恢复保活；主任务 PID 和四个工作 GPU 进程保持不变，STAR 仍从 22,027 推进到 22,048 条。阶段切换期间也实测了借卡任务继续执行。CPU、内存与磁盘仍为共享资源，详见 [借卡约束与命令](SPARE_GPU_TASKS.md)。

77 项回归测试通过。正式任务已从原断点恢复并继续运行，运行级错误为 0。进度现在在加载期间显示已恢复数量，ETA 不再把冷加载计入样本处理速率。

测量证据：[转换](../reproduction/evidence/parallel_text_benchmark.json)、[筛选](../reproduction/evidence/parallel_filter_benchmark.json)、[真实部分断点恢复](../reproduction/evidence/parallel_text_resume.json)、[评测 smoke](../reproduction/evidence/parallel_evaluation_smoke.json)、[借卡隔离](../reproduction/evidence/spare_gpu_isolation.json)、[估时采样](../reproduction/evidence/parallel_eta_snapshot.json)。估时采样时 STAR 为 22,232/45,731，最近 128 条约 1.49 秒/完成记录；AGQA 随机 64 条约 1.28 秒/完成记录。两者点估计约 9.7 小时、8.9 小时；题型与视频不同，不能把抽样吞吐当作完成时间保证。全量训练和全量评测尚未结束。

---

# 2026-11-09：四卡独立样本并行

全量入口默认 `AOTD_CONSTRUCTION_WORKERS=4`，以 spawn 启动四个隔离进程，每卡常驻完整教师栈、同时处理一条样本。主进程负责唯一 journal 写入，保留逐条 fsync、运行级错误停机和保活恢复机制。教师、提示词、贪心解码、拒绝与质量筛选规则未变。并行改变完成顺序，不改变样本 ID；最多四条未提交任务会在中断后重做。

已完成 74 项回归（其中 5 项离线 processor 测试单独配置模型路径执行）。新增测试覆盖乱序完成、唯一落盘、关闭后续跑、子进程异常及空队列快路径。

真实对照使用相同 40 条 QA（32 条 STAR、8 条 AGQA），串行与并行均得到 16 条执行成功、24 条拒绝、0 条运行级错误、2 条答案正确。逐条比较生成程序、答案、状态与正确性，没有差异。模型 ready 至最后提交耗时 227.71 秒 → 71.95 秒，吞吐约 **3.16 倍**；包含冷加载的 manifest 至最后提交耗时 264.97 秒 → 244.51 秒，短任务总耗时仅约 **1.08 倍**。这些是该测试集测量结果，不代表整个数据集恒定加速比，也不是最终 CoT 接受率。

暂停时 NExT-QA 已完成全部 34,132 条构建，STAR 已持久化 21,612 条。保留 journal 前缀 SHA256，并通过精确代码指纹白名单迁移恢复。完整机制见 [并行构建说明](PARALLEL_CONSTRUCTION.md)，对照原始汇总见 [测量记录](../reproduction/evidence/parallel_construction_benchmark.json)。正式全量实际推进 47 条至 21,659 条后，以 SIGTERM 停止 guard；退出码 143，原 8 个保活 worker 均恢复。相同全量命令再次启动，跳过全部已完成记录，继续新增 53 条至 21,712 条。两次 journal 前缀 SHA256 校验通过、无重复终态记录、运行级错误为 0；合计新增 100 条后保留任务运行。见 [恢复证据](../reproduction/evidence/parallel_construction_restart.json)。这验证了本轮测试范围内的恢复机制，不等于宣称全量训练评测已经完成。

---

# 开发记录：制数边界修复与断点恢复

日期：2026-11-07。保留五源协议、133,654 条有效 QA、4 张工作卡与 4 张保活卡、默认 30 秒进度间隔。本次不修改数据来源、训练目标或解释筛选标准。

## 1. 实际失败与影响

### 本轮：第 2,564 个样本触发检测器文本越界

最新 full 在 NExT-QA 制数阶段、样本 `4563040340_0`（原始行号 2,563，从 0 开始）处失败。重启后已从 2,208 个样本推进到 2,564 个；失败时为 2,053 条执行成功、510 条拒绝、1 条运行级错误，628 条答案正确，尚未进入学生训练。

生成程序将 `Query_Actions` 返回的长段自然语言传给 `Count`。OWLv2 加上检测提示前缀后得到 **97 个 token**，实际加载模型的文本位置编码上限只有 **16**；原适配器仅要求 padding，没有启用 truncation，导致文本编码器内部张量维度不匹配。安装的 Transformers `4.49.0` 实现与[该版本上游 processor 源码](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/owlv2/processing_owlv2.py#L45-L55)一致：默认 padding 不会自动截断超长输入。

这是视觉工具输入边界缺失，不能把该模型异常改成普通拒绝来绕过修复，也不应删除这条原始 QA。退出后原 8 卡保活已恢复。上轮只新增两条真实样本的有界验证不足以覆盖此输入组合，不能作为长时间 full 稳定性的证据。

本次恢复必须原样保留旧制数 journal 的 **3,466,004 字节**前缀，SHA256 为 `1645c76b613b2c0e3f225b99e1cf9ceb255f05f3443a88e12f26f0d7df9f0ece`。失败样本在修复后重试，成功和正常拒绝记录继续复用；旧日志与错误记录保留。

### 上轮历史：生成程序局部变量错误

上一轮正式制数在第 2,206 条持久记录处停止。样本 `9215666935_0` 的生成程序先使用 `actions`，随后才给它赋值，触发 `UnboundLocalError`。异常最深位置在生成程序内部，原驱动却将其分类为运行级错误。

当时持久索引统计为 1,772 条执行成功、433 条正常拒绝、1 条运行级错误；546 条答案正确。已完成记录和原始日志全部保留。上一轮受测范围内 OOM/OOM-kill 计数增量为 0，退出后原 8 卡保活恢复；该历史故障不能归因为内存耗尽。

## 2. 本轮修复与无损制数恢复

OWLv2 现在从实际加载模型的 `text_config.max_position_embeddings` 读取上限，显式传入 `truncation=True`、`padding=max_length` 和对应 `max_length`。长描述保留结束 token，多个检测对象采用相同上限，短描述编码保持一致；没有捕获或吞掉真实模型/CUDA `RuntimeError`。

同时补齐视觉工具的相邻输入边界：空视频、倒置或空时间区间，以及非有限、倒置、无面积或完全位于画面外的裁剪框，在调用后续模型或 resize 前明确报错；有效裁剪仍保留原有图像处理。这里拒绝的是无法执行的视觉选择，原始 QA 和视频均保留。

制数来源指纹扩展为驱动、IO、实际 Agent Python 源文件、工具提示词及配置的 SHA256。`bounded-visual-tool-inputs-v1` 迁移政策列明审核过的旧版本和新版本指纹，其他输入、模型和提示词仍须一致；旧 manifest 备份后才记录迁移。它不会允许任意工具改动复用旧结果。失败结果现在也保存当次生成程序和部分执行轨迹，便于复现模型调用错误。

- 只将回溯最深位置确实来自生成程序的相应 Python 数据错误转换为拒绝样本。
- 模型、视觉工具和内存等基础设施错误继续中止阶段并保留失败信息，重启时自动重试运行级错误。
- 正常成功与正常拒绝记录默认不重做；`--retry-errors` 是显式额外重试控制。
- 只对兼容政策中精确列明的旧/新代码与工具来源指纹执行迁移，其他变化拒绝混用。
- 保存原 manifest 到 `manifest_history/` 并记录 `code_migrations`；原始 JSONL 前缀不修改，只追加旧错误的新分类。

兼容政策见 [construction_resume_compatibility.json](../reproduction/metadata/construction_resume_compatibility.json)。它是针对这次边界修复的明确迁移，不是关闭来源指纹校验。

正式训练池也增加发布事务：在暂存目录完成 QA、CoT、两份 YAML 和 manifest，校验全部文件哈希并 fsync 后，按固定顺序发布，manifest 最后发布。若中断发生在部分文件发布之后，同一输入再次启动会校验并补齐尚未发布的文件；源数据变化或已发布文件冲突会保留现场并停止。full 不再因为已有一份 YAML 就误判整个训练池已经准备完成。

## 3. 三层 checkpoint

| 阶段 | 保存内容 | 重启行为 |
|---|---|---|
| 制数 | 每条结果、执行轨迹、成功/拒绝/错误分类及来源指纹 | 恢复 JSONL 尾部，按最新记录续接，重试运行级错误 |
| 学生训练 | 模型、processor、优化器、调度器、各 rank RNG、Trainer/DeepSpeed 状态 | 自动选择完整且兼容的 checkpoint，恢复 global step、采样位置和训练状态 |
| 评测 | 每道题的持久结果与模型/CSV/设置指纹 | 校验完整前缀，只处理未完成题目；已完成结果核验后跳过 |

正式训练默认每 **100 个 optimizer step** 保存一次，保留最近 **2 个 checkpoint**，通过 `AOTD_SAVE_STEPS` 和 `AOTD_SAVE_TOTAL_LIMIT` 配置。最后一步即使不在保存间隔上也保存训练状态，之后再导出最终模型。

上一轮短训练实测一份完整四卡 checkpoint 约 **100 GB**；每个训练阶段保留两份，两个阶段合计约 400 GB，另需容纳最终权重和数据。保存完整优化器状态存在明显 I/O 开销，保存间隔可以按恢复损失容忍度调整。

`checkpoint-N/aotd_checkpoint.json` 在各 rank 完整保存并检查后原子发布。只有推理权重、缺 optimizer/RNG、缺分片或没有完成标记的目录不能作为完整 checkpoint。最新目录不完整时可回退到较早完整目录，所有兼容性检查仍必须通过。

`training_provenance.json` 绑定训练输入、Base 权重身份和训练配置；`training_complete.json` 表示完整训练阶段及最终文件已验证。中断发生在最后 checkpoint 之后、最终导出完成之前时，应从已保存状态恢复导出，不能多训练一步。

本轮补齐首次保存尚未提交的恢复路径：仅在 `training_provenance.json` 完全匹配、所有候选 checkpoint 都没有完成标记时，保留原目录到 `checkpoint_history/`，记录 `checkpoint_restart.json`，明确从 **step 0 重新训练**。隔离过程自身中断也能恢复，四个 rank 通过文件锁串行选择。已有完整 checkpoint 仍优先恢复；完成标记损坏、已提交文件缺失或来源不一致不能冒充未完成保存。

本次还核对了固定训练库版本的 epoch 边界：默认梯度累积调度会向下取整，直接使用 133,654 条 QA 时可能漏掉最后 22 条。正式 full 因此显式启用 `--pad_to_effective_batch`，在训练视图中确定性重复最少数量的真实记录，使长度整除有效 batch 48。Instruct 为 133,654 → 133,680（增加 26 个真实记录的重复采样），共 2,785 个完整更新；AoTD 根据实际 QA+CoT 总长度独立计算。原始训练 JSON 不改写，重复索引、ID 摘要和计数写入 provenance。固定步数的 smoke 不启用该策略，避免混淆验证记录数。

## 4. 一键启动与同目录恢复

```bash
bash run_aotd_full.sh
```

默认首次启动创建受管理的运行目录并记录 `runtime/latest_all_run.json`；后续同一命令自动复用该目录。日志追加写入，不覆盖早先的错误和训练记录。项目锁拒绝同一目录重复启动。

```bash
bash run_aotd_full.sh --resume /absolute/project/results/full_RUN
bash run_aotd_full.sh --new-run
bash run_aotd_full.sh --eval-only
bash run_aotd_full.sh --check
```

`--resume` 选择兼容的受管理运行，`--new-run` 显式创建新的训练/评测结果目录。制数轨迹仍独立持久保存，`--new-run` 不会删除已有制数记录。旧版本没有 `run_state.json` 的结果目录不会被冒充为新的受管理目录；其制数记录可以按上述审核规则继续使用。

输入或训练配置不兼容时，程序保留已有文件并明确失败。不会静默清空旧权重或把旧阶段结果当作新实验。`--check` 使用独立检查目录，不替换正在使用的运行指针。

## 5. 验证记录

### 本轮边界与发布回归

目标运行环境中隐藏 CUDA 后，最终 **69 项 CPU 回归全部通过，耗时 9.391 秒**。其中包括 5 项真实离线 OWLv2 processor 测试、7 项真实张量时间选择/裁剪边界测试、5 项训练池发布与恢复测试、6 项首次 checkpoint 保存中断/并发恢复新测试，以及已有制数、来源迁移、checkpoint、评测恢复和运行目录测试。

OWLv2 processor 测试覆盖超长描述、长短混合的多个对象、短描述编码不变、模型上限优先于错误的 tokenizer 默认值，以及模型异常注入后不被重新分类或重试。张量边界测试使用隔离的测试张量，发布测试注入中断并核对恢复后的完整文件和哈希；这些测试夹具均不进入训练数据。

| 本轮实际运行验证 | 当前状态 |
|---|---|
| 精确失败对象查询的真实 OWLv2 前向，核对 97→16 token、结束 token 和有限输出 | PASS；真实模型 logits/boxes 均为有限值，结束 token 保留；退出后原 8 个保活 worker 恢复 |
| 从正式 full 入口恢复，重试失败样本并持续越过失败位置 | 首段 PASS：2,564 → 3,004，新增 440 个唯一样本；构建阶段连续运行约 23 分钟，新增运行级错误 0 |
| 原 journal 前缀 SHA、保活进程与新增错误核验 | 原 3,466,004 字节前缀 SHA 不变；4 工作设备 + 4 个就绪保活 worker；受控中断退出 143 后原 8 个同身份 worker 全部活动，重复 worker 为 0 |
| 同一 full 命令中断后续跑 | PASS：从 3,004 条续接至 3,075 条，新增 71 条、运行级错误 0、终态样本重复执行 0；完整重启前 4,159,991 字节前缀 SHA 保持 |

截至 `2026-11-07T02:16:16Z`，相对本轮原基线共新增 **511** 个唯一样本，当前索引为 **3,075** 条（执行成功 2,436、拒绝 639、运行级错误 0），正式 full 仍在后台继续制数，4 个工作设备和 4 个就绪保活 worker 正常。此观察不是完整三源 CoT、正式训练或完整评测完成标记。

原失败样本重试后不再触发检测器的维度错误，但生成程序随后对整数调用 `.items()`，因此按生成程序错误记录为 `rejected`。这不是合格 CoT，也没有伪造程序执行成功。构建阶段已有其他新样本成功执行，原始 QA 不删除。

仓库中的 [真实模型诊断入口](../tools/validate_owlv2_replay.py)已在正常保活调度下实测，使用保存的原对象查询和原视频帧。原查询 SHA 为 `98a1e7c407833aebfe8340558bdfe011fb08888fe42aaeae3b1e266a4f300520`，97→16 token、EOS 和模型输出有限性均通过。诊断 JSON 和 full 观察摘要见 [tool_boundary_validation.json](../reproduction/evidence/tool_boundary_validation.json)。

本轮资源观察是瞬时采样：在约 100/250/440 个新增样本处，构建进程 RSS 约为 5.03/5.41/5.51 GiB，所在 cgroup 内存约为 13.12/13.95/14.25 GiB，上限为 1,760 GiB，OOM/OOM-kill 计数为 0。没有内核 `memory.peak` 文件，不能把采样值称为真实峰值或据此保证剩余全部数据的资源使用。

### 上一轮历史：完整 checkpoint 和评测续跑

以下结果以 [checkpoint_validation.json](../reproduction/evidence/checkpoint_validation.json) 中实际通过的项目为准，均是上一轮的短训练及中断恢复验证，不代表本轮 full 已完成。

上一轮实际 GPU 验证覆盖：step 1 完整 checkpoint 后中断，恢复至 step 2；核对四 rank optimizer/RNG、global step、loss 和保存后的模型重载；完成评测前缀后中断再续跑，验证不重做已完成题；对当时正式制数的 2,206 条前缀做 SHA 核对和有界续跑。所有 GPU 验证退出后核对原 8 个保活进程身份与无额外工作进程。

上一轮已完成的实际训练验证：step 1 checkpoint 原子发布后 SIGTERM 中断（退出 143），随后恢复至 step 2，loss 为 `15.7559/15.8379`，日志步数恰为 `[1,2]`；四 rank optimizer/RNG 文件齐全，最终模型与 processor 实际重载评测通过。正常和中断退出均恢复原 8 个保活进程，无重复工作进程。

上一轮已完成的恢复场景：

- 最后 checkpoint 已达到 step 2 而最终标记缺失时，恢复导出新增 optimizer step 为 0，最终权重 SHA 不变；重复已完成阶段不改写权重。
- 两题真实评测在第一题持久落盘后中断，续跑只补算第二题，无重复 ID。隐藏 CUDA 后可以复用已完成结果；最终 JSON 缺失时也能从完整持久 JSONL 重建统计，无需重新推理。
- 原正式制数的 2,206 条、2,928,975 字节前缀 SHA 保持不变；追加一次历史错误重分类并新增两条真实结果，当时索引为 2,208 个样本，新增运行级错误为 0。验证后已停止有界任务，原 8 卡保活恢复。之后的正式 full 在 2,564 个样本处遇到本文第 1 节的另一处工具错误。
- 18 项原回归、5 项错误/迁移回归、12 项 checkpoint/补齐回归、4 项评测恢复回归、6 项运行目录回归合计 **45 项通过**；另验证了实际 HF Dataset 的训练视图补齐，原文件不变。

本轮验证不等于完整一轮训练或完整评测已经完成，具体完成状态必须以结果标记与逐题完整性核验为准。

## 6. 集成方式

集成分支为 `integration/offline-reproduction`。原 [PR #2](https://github.com/GentaLee/AoTD/pull/2) 与 checkpoint 修复 [PR #3](https://github.com/GentaLee/AoTD/pull/3) 已合入；本轮视觉工具边界及恢复修复见 [PR #4](https://github.com/GentaLee/AoTD/pull/4)，目标为同一集成分支。未直接合入 `main`。
