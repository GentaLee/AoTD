# 开发记录：故障恢复与完整 checkpoint

日期：2026-11-07。保留五源协议、133,654 条有效 QA、4 张工作卡与 4 张保活卡、默认 30 秒进度间隔。本次不修改数据来源、训练目标或解释筛选标准。

## 1. 实际失败与影响

正式制数在第 2,206 条持久记录处停止。样本 `9215666935_0` 的生成程序先使用 `actions`，随后才给它赋值，触发 `UnboundLocalError`。异常最深位置在生成程序内部，原驱动却将其分类为运行级错误。

当时持久索引统计为 1,772 条执行成功、433 条正常拒绝、1 条运行级错误；546 条答案正确。已完成记录和原始日志全部保留。OOM/OOM-kill 计数增量为 0，退出后原 8 卡保活恢复。本次故障不能归因为内存耗尽。

## 2. 故障分类与无损制数恢复

- 只将回溯最深位置确实来自生成程序的相应 Python 数据错误转换为拒绝样本。
- 模型、视觉工具和内存等基础设施错误继续中止阶段并保留失败信息，重启时自动重试运行级错误。
- 正常成功与正常拒绝记录默认不重做；`--retry-errors` 是显式额外重试控制。
- 对本次已审核的旧/新 driver 与 IO SHA 对执行兼容迁移；其他输入、模型、提示词或代码变化仍拒绝混用。
- 保存原 manifest 到 `manifest_history/` 并记录 `code_migrations`；原始 JSONL 前缀不修改，只追加旧错误的新分类。

兼容政策见 [construction_resume_compatibility.json](../reproduction/metadata/construction_resume_compatibility.json)。它是针对这次边界修复的明确迁移，不是关闭来源指纹校验。

## 3. 三层 checkpoint

| 阶段 | 保存内容 | 重启行为 |
|---|---|---|
| 制数 | 每条结果、执行轨迹、成功/拒绝/错误分类及来源指纹 | 恢复 JSONL 尾部，按最新记录续接，重试运行级错误 |
| 学生训练 | 模型、processor、优化器、调度器、各 rank RNG、Trainer/DeepSpeed 状态 | 自动选择完整且兼容的 checkpoint，恢复 global step、采样位置和训练状态 |
| 评测 | 每道题的持久结果与模型/CSV/设置指纹 | 校验完整前缀，只处理未完成题目；已完成结果核验后跳过 |

正式训练默认每 **100 个 optimizer step** 保存一次，保留最近 **2 个 checkpoint**，通过 `AOTD_SAVE_STEPS` 和 `AOTD_SAVE_TOTAL_LIMIT` 配置。最后一步即使不在保存间隔上也保存训练状态，之后再导出最终模型。

本次实测一份完整四卡 checkpoint 约 **100 GB**；每个训练阶段保留两份，两个阶段合计约 400 GB，另需容纳最终权重和数据。保存完整优化器状态存在明显 I/O 开销，保存间隔可以按恢复损失容忍度调整。

`checkpoint-N/aotd_checkpoint.json` 在各 rank 完整保存并检查后原子发布。只有推理权重、缺 optimizer/RNG、缺分片或没有完成标记的目录不能作为完整 checkpoint。最新目录不完整时可回退到较早完整目录，所有兼容性检查仍必须通过。

`training_provenance.json` 绑定训练输入、Base 权重身份和训练配置；`training_complete.json` 表示完整训练阶段及最终文件已验证。中断发生在最后 checkpoint 之后、最终导出完成之前时，应从已保存状态恢复导出，不能多训练一步。

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

故障边界的 5 项新回归、原有 18 项数据/制数回归，以及运行目录选择/隔离的 6 项回归已通过。完整 checkpoint 和评测续跑的结果以 [checkpoint_validation.json](../reproduction/evidence/checkpoint_validation.json) 中实际通过的项目为准。

实际 GPU 验证要求：step 1 完整 checkpoint 后中断，恢复至 step 2；核对四 rank optimizer/RNG、global step、loss 和保存后的模型重载；完成评测前缀后中断再续跑，验证不重做已完成题；对正式制数的 2,206 条前缀做 SHA 核对和有界续跑。所有 GPU 验证退出后核对原 8 个保活进程身份与无额外工作进程。

已完成的实际训练验证：step 1 checkpoint 原子发布后 SIGTERM 中断（退出 143），随后恢复至 step 2，loss 为 `15.7559/15.8379`，日志步数恰为 `[1,2]`；四 rank optimizer/RNG 文件齐全，最终模型与 processor 实际重载评测通过。正常和中断退出均恢复原 8 个保活进程，无重复工作进程。

全部恢复场景已完成：

- 最后 checkpoint 已达到 step 2 而最终标记缺失时，恢复导出新增 optimizer step 为 0，最终权重 SHA 不变；重复已完成阶段不改写权重。
- 两题真实评测在第一题持久落盘后中断，续跑只补算第二题，无重复 ID。隐藏 CUDA 后可以复用已完成结果；最终 JSON 缺失时也能从完整持久 JSONL 重建统计，无需重新推理。
- 原正式制数的 2,206 条、2,928,975 字节前缀 SHA 保持不变；追加一次历史错误重分类并新增两条真实结果，最终索引为 2,208 个样本，新增运行级错误为 0。验证后已停止有界任务，原 8 卡保活恢复，没有启动长期全量。
- 18 项原回归、5 项错误/迁移回归、12 项 checkpoint/补齐回归、4 项评测恢复回归、6 项运行目录回归合计 **45 项通过**；另验证了实际 HF Dataset 的训练视图补齐，原文件不变。

本轮验证不等于完整一轮训练或完整评测已经完成，具体完成状态必须以结果标记与逐题完整性核验为准。

## 6. 集成方式

新建集成分支 `integration/offline-reproduction`。原 [PR #2](https://github.com/GentaLee/AoTD/pull/2) 已改为指向该分支并合入；本次修复使用 [PR #3](https://github.com/GentaLee/AoTD/pull/3) 指向同一分支。两个变更在集成分支汇合，未直接合入 `main`。
