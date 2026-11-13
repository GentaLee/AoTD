# 最新状态 / Latest status

2026-11-13：第二轮全量已完成，`complete` / exit 0 / `PASS`。133,654 条 QA、32,042 条 CoT；AoTD 训练 3,452 步，四组各完整评测 4,996 题。AoTD 71.00%，Instruct 70.60%，作者模型 77.62%。Instruct 为经一致性校验复用的第一轮对照。完成日志确认工作组保活恢复，独立借卡组不受影响。

The second full run completed successfully. It produced 32,042 rationales, trained AoTD for 3,452 updates and evaluated all four models on 4,996 questions each. AoTD achieved 71.00%, versus 70.60% for the validated reused Instruct control and 77.62% for the author checkpoint. Completion-time logs confirm work-group keepalive restoration.

最新数据及局限见[第二轮结果](FULL_V2_RESULTS_20261113.md)，操作与交付范围见[复现索引](REPRODUCTION_HANDOFF.md)。下文是第一轮及早期 smoke 的历史记录，其中“尚无”“当前仍在”等描述均指当时状态，不代表第二轮状态。

See the [latest results](FULL_V2_RESULTS_20261113.md) and [reproduction handoff](REPRODUCTION_HANDOFF.md). Everything below is historical first-run/smoke evidence; pending-work statements describe those observation times.

---

# 五源离线复现状态与运行说明

第一轮状态日期：2026-11-10。第一轮完整运行已结束，结果以[结果报告](FULL_RUN_RESULTS_20261110.md)为准，开发汇总见[开发报告](FULL_RUN_DEVELOPMENT_20261110.md)。下方旧 smoke 与故障观察保留作历史证据。此文区分已经验证的行为、入口具备的功能和仍需完成的全量实验。部署地址、站点名称、设备型号及凭证不保存到仓库，软件版本可以记录。

**当前范围排除 ANetQA，使用 NExT-QA、STAR、AGQA、CLEVRER、EgoQA。教师模型齐全，真实 Agent 小样本链路、四卡有效 batch 48 短训练及 checkpoint 保存/重载已经通过；五源完整 CoT、Instruct/AoTD 两次各一轮训练和四组完整 val 评测均已完成。** EgoQA 已完成全部 7,813 个标注视频的 32 帧解码审计：7,791 个有效、16 个缺失、6 个无法解码。按明确授权的固定清单排除这 22 条媒体异常记录，原始标注保持不变；当前五源可用 QA 合计 133,654 条。

**第一轮最终状态：** `complete` / exit 0 / `PASS`；133,654 条 QA、4,399 条 CoT，Instruct 2,785 步、AoTD 2,877 步；四组各完成 4,996 题。Base 62.47%、Instruct 70.60%、AoTD 70.64%、作者权重 77.62%。八卡保活已恢复。

## 1. 协议与数据范围

配置由 [sources.json](../config/construction/sources.json) 驱动，协议 ID 为 `five-source-valid-media-no-anetqa-v1`。

| 训练源 | QA 数 | 制作 CoT | 当前边界 |
|---|---:|---|---|
| NExT-QA train | 34,132 | 是 | 已归一化，3,870 视频，缺失 0 |
| STAR train | 45,731 | 是 | 已归一化，16,206 个唯一时间窗覆盖全部问题，缺失 0 |
| AGQA train | 25,000 | 是 | 确定性重建抽样，5,538 视频，缺失 0 |
| CLEVRER train | 21,000 | 否 | 原生逐候选二分类标签，8,877 视频，缺失 0 |
| EgoQA train | 7,791 | 否 | 原始 7,813 条中显式排除 16 条媒体缺失、6 条解码失败，其余全部通过 32 帧解码 |
| 五源合计 | **133,654** | 三源需 CoT | 完整训练已完成 |

EgoQA 的媒体异常不是传输截断：原始归档大小与发布方 SHA256 一致，缺失 16 个精确路径；另外 6 个 MP4 各仅 262 字节，实际解码失败。已检查同源镜像及补充分片目录，未找到可验证的替代片段。采用[固定 22 条排除清单](../reproduction/metadata/egoqa_media_exclusions.json)，保留原生标注 SHA256、逐项原因与有效媒体名单哈希，并保存[完整解码审计](../reproduction/evidence/egoqa_media_decode_audit.json)。这是一项公开记录的数据协议变化，不是把缺失样本静默跳过，也没有使用其他片段代替。

ANetQA 不再是默认配置的前置依赖，混入该来源或其他未启用来源的训练记录会被拒绝。完整 preflight 按各启用来源的最低真实 QA 覆盖检查，并检查 Instruct/AoTD 使用相同 QA 文件、CoT 只来自启用来源且能够匹配源 QA、媒体全部存在。不会沿用六源的固定总数门槛，也不会接受少量 smoke 记录作为全量输入。

原作者完整处理后 QA/CoT、抽样 ID 与程序生成 few-shot 示例未公开，因此本工程采用明确标记的确定性重建。排除一个来源后更不能将本次结果称为原论文完整训练池的严格复现。

NExT-QA 原标注存在同一个 `video_qid=3617262562_2` 对应不同问题的重复 ID，归一化保留两题并为第二题增加原始行号后缀。CLEVRER 某些问题有多个正确候选，采用原生逐候选 Yes/No 标签，不能强制转换成单个正确选项。

## 2. 教师和学生权重

**全部教师及辅助编码器均已齐备，且完成实际加载与前向验证。** 两份本地 Llama 均为 BF16，四个分片均与发布方 SHA256 对齐，目标侧再次校验通过。历史下载权限错误不再是模型推理阻塞。

| 角色 | 相对路径 | 公开来源 |
|---|---|---|
| 程序规划 | `models/teachers/DeepSeek-Coder-6.7B-Instruct` | [DeepSeek](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) |
| 视频专家 | `models/teachers/LLaVA-Video-7B-DPO` | [LLaVA DPO](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-DPO-hf) |
| 开放词汇检测 | `models/teachers/OWLv2` | [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) |
| 时间定位 | `models/teachers/UniVTG` | [官方模型目录](https://drive.google.com/drive/folders/1-eGata6ZPV0A1BBsZpYyIooos9yjMx2f) |
| 时间定位视觉编码器 | `models/teachers/CLIP` | [ViT-B/32 权重](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) |
| 答案选择 | `models/teachers/Meta-Llama-3-8B-Instruct` | [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| 轨迹转换与解释筛选 | `models/teachers/Llama-3.1-8B-Instruct` | [Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| 学生 Base | `models/LLaVA-7B-base` | [LLaVA Base](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) |
| 作者发布模型 | `models/AoTD-7B-author` | [AoTD-7B](https://huggingface.co/Zhengrongzz/AoTD-7B) |

来源 revision、文件大小及哈希记录在[验证快照](../reproduction/evidence/verified_status.json)和[Llama 发布方清单](../reproduction/metadata/llama_publisher_hashes.json)。不在 Git 中分发模型二进制。

## 3. 历史 smoke 与修复证据（最终结果见独立报告）

已验证的软件环境包括 PyTorch `2.11.0a0+eb65b36914.nv26.02`、CUDA `13.1`、Transformers `4.49.0`、DeepSpeed `0.15.4`、NumPy `1.26.4`，使用 BF16/SDPA。版本记录不意味着所有其他镜像组合都已验证。

| 验证 | 已测结果 | 适用边界 |
|---|---|---|
| 本轮工具边界和恢复回归 | 69 项 CPU 测试通过，9.391 秒；真实 detector 长查询前向通过，97→16 token、结束 token 和有限输出已核验 | 首次未提交 checkpoint 从零重训明确记录，与完整断点恢复区分 |
| 本轮 full 观察与中断 | 2,564 → 3,004 后受控中断，退出 143 后原 8 个保活 worker 同身份恢复；同目录续接至 3,075，新增运行级错误与重复执行均为 0，两段旧前缀 SHA 保持 | 当前仍在全量制数；不是全量训练完成 |
| 教师前向 | 程序生成、视频问答、目标检测、时间定位、答案选择、转换和筛选均通过 | 有限小样本功能验证 |
| 真实 Agent 链路 | 4 道 train QA：3 次程序执行成功、1 次非代码回复被拒；1 条答对并完成转换、筛选和 CoT 导出 | 不是全量准确率或合格率 |
| 正式 batch 配置短训练 | 4 卡、microbatch 1、累积 12、有效 batch 48、ZeRO-2、BF16；2 步 loss `15.7559` / `15.8379`，有限梯度 | 96 条受控重复 QA，只验证训练配置 |
| 作者权重和保存重载 | 作者评测 2 题；短训练后保存 checkpoint/processor，重载评测 1 题，通过 | 不是完整评测指标 |
| 保活恢复 | 最终 smoke、注入错误退出 23、SIGTERM 退出 143、SIGKILL watchdog 路径均通过；原 8 个相同 PID 恢复、重复进程 0 | 受测范围内 OOM 计数增量 0；整体节点退出不在脚本恢复范围 |
| 完整评测输入检查 | val 4,996 问题、570 视频的输入覆盖检查通过 | 尚无完整推理分数 |
| 新布局回归 | 两个独立环境的 10 项原始回归通过；五源覆盖、排除来源、时间节流和最终即时输出行为检查通过 | 最终进度版本已重新执行实际训练、评测和保存重载 smoke |

[历史证据字段](../reproduction/evidence/verified_status.json)、[仓库布局检查](../reproduction/evidence/repository_integration.json)随代码保存。最终复测与异常恢复记录保存在 `reproduction/evidence/current_validation.json`，仅实际完成的项目标记 PASS。原始逐题日志、轨迹、训练数值和进程审计保留在挂载卷，公开证据只导出不含站点信息的必要字段。

最新失败样本为 `4563040340_0`：工具接收到的长对象描述经 OWLv2 tokenizer 编码为 97 token，超过加载模型的 16 token 上限。Transformers `4.49.0` 的 [processor 默认值](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/owlv2/processing_owlv2.py#L45-L55)仅设置 padding，原适配器未显式截断，因此异常发生在真实模型文本编码器内。现改为读取模型实际位置编码上限并保留结束 token，同时在模型调用前拒绝空时间选择和无效裁剪。这次修复不删除原始 QA，也不把真实模型 `RuntimeError` 静默改为普通拒绝。

训练修正保留原版 padding-only label collator 的监督范围，未暗中改成 answer-only loss。设置 `model.accepts_loss_kwargs=False`，避免 Transformers 对 `**kwargs` 的能力推断导致 microbatch mean CE 在累积时缩放错误；同时记录真实运行引擎、world size、ZeRO stage、有效 batch 和逐步数值。

## 4. full 一键入口

在已经配好资源的持久化 checkout 中运行：

```bash
bash run_aotd_full.sh
```

完整流程依次为：

1. 若缺少正式训练记录，先检查五源标注与媒体，再按配置对 NExT-QA、STAR、AGQA 执行真实 Agent 程序、视觉工具、正确答案门控、轨迹转换与解释筛选；CLEVRER/EgoQA 提供原生 QA。
2. 导出完整五源 QA/CoT，生成 Instruct/AoTD YAML，检查来源覆盖、唯一 ID、匹配关系和全部媒体。缺失资源或没有合格 CoT 会中止，不会制造训练标签。
3. Instruct（QA-only）与 AoTD（QA+CoT）分别从相同 Base 独立初始化，各训练完整 1 epoch，默认 seed 42、4 张工作卡、microbatch 1、梯度累积 12、有效 batch 48、ZeRO-2、BF16/SDPA。没有 smoke 的步数/样本上限。
4. 对作者/Base/Instruct/AoTD 分别评测完整 NExT-QA val 4,996 题。设置 `AOTD_EVAL_SPLIT=test` 可改为 8,564 题，但需要对应媒体全部齐备，不得混报两种划分。
5. 完整覆盖每组评测问题后才生成最终 `full_summary.json`。

只检查资源：`bash run_aotd_full.sh --check`。只做作者/Base 完整评测：`bash run_aotd_full.sh --eval-only`，该模式不会训练自己的 Instruct/AoTD。

制数可单独运行 `bash run_aotd_build_data.sh`，检查模式为 `--check`。指定 `--dataset NExT-QA` 可先运行单源，但默认 full 仍要求全部五源覆盖，不能把单源制数当作全量复现。

## 5. GPU 保活、进度与中断

训练使用 4 张工作卡；构建、转换和筛选默认四进程并行，四模型评测各占一张工作卡并行。其余 4 张卡为独立保活/借用组，详见 [借卡说明](SPARE_GPU_TASKS.md)。保活只是运行保障，不能将四卡训练报告为八卡训练。

原保活进程树由 guard 暂停和恢复，独立 watchdog 处理父进程异常；正常退出、错误与可处理信号退出都应恢复原保活。进程被系统强制终止后的恢复依赖 watchdog 仍在运行，宿主机/容器整体退出不能由同容器脚本保证恢复。

默认 `AOTD_PROGRESS_INTERVAL_SECONDS=30`，允许 10–60 秒。阶段完成立即输出，不必等下个间隔。训练展示 step、总步数、百分比、已用时间、预计剩余时间和 loss；评测展示题目进度与预计剩余时间。逐步数值、逐题 JSONL 和失败信息完整保留。默认不逐题、逐 token 刷屏，`AOTD_VERBOSE=1` 可查看评测细节。

制数 JSONL 逐条落盘，可恢复中断尾部并按输入、提示词、驱动、IO、实际工具源文件与模型指纹检查兼容性。针对精确审核过的版本指纹提供保留原始前缀和旧 manifest 的迁移，不允许任意改动混用旧结果；失败结果保留当次程序与部分轨迹。此次待恢复的旧 journal 前缀为 3,466,004 字节，SHA256 `1645c76b613b2c0e3f225b99e1cf9ceb255f05f3443a88e12f26f0d7df9f0ece`。

正式训练池先在暂存目录生成并校验，manifest 最后发布；中断后按原输入和文件哈希补齐发布，不会因只存在部分 JSON/YAML 就误判为完整训练输入。学生训练默认每 100 步保存完整 checkpoint，保留最近 2 个，恢复 optimizer/scheduler/RNG/Trainer/DeepSpeed 状态；评测按已完成题目的持久前缀续跑。同一 full 命令默认复用最新受管理运行，`--new-run` 创建新训练/评测目录，`--resume RUN_DIR` 指定目录。详细恢复边界和实测结果见 [开发记录](DEVELOPMENT_LOG.md)。

## 6. 目录、配置与结果

| 相对位置 | 用途 |
|---|---|
| `runtime/site.env` | 站点专属环境变量，不提交 Git |
| `models/`、`data/`、`downloads/` | 模型、媒体与离线依赖，全部保存在挂载卷 |
| `data/construction/` | 逐源归一化 QA、Agent 轨迹、转换与筛选记录 |
| `data/five_source_sources/` | 合并后的五源 QA/CoT 和来源 provenance |
| `data/five_source_train/` | 正式训练 JSON、manifest、Instruct/AoTD YAML |
| `results/full_<UTC时间>_<PID>/` | `run.log`、两组 checkpoint、逐题 JSONL/JSON、最终统计 |
| `results/build_<UTC时间>_<PID>/` | 制数阶段运行日志与状态 |

站点配置示意：

```bash
# 写入不提交 Git 的 runtime/site.env，并替换为本地持久化路径
export AOTD_KEEPALIVE_SCRIPT=/absolute/persistent/path/to/keep_alive/main.py
export AOTD_PROGRESS_INTERVAL_SECONDS=30
```

已有离线资源通过 `bash use_local_assets.sh /absolute/path/to/bundle --reuse-env` 关联。仅在兼容的同节点环境中复用 env；新环境通过 `setup_offline.sh` 与 `setup_agent_offline.sh` 安装本地 wheel。跨主机同步需要外部设置 `AOTD_WORKER_HOST`、`AOTD_WORKER_ROOT`，必要时设置 `AOTD_WORKER_PLATFORM_PYTHON`，再运行 `sync_to_worker.sh`。所有新增运行数据、缓存与临时文件都应位于持久化挂载卷。

程序运行不写 Markdown 进度日志，本文是供审查的状态说明。凭证必须位于仓库外的私有目录，不能上传到 issue、PR、日志或下载清单。

## 7. 数据公开入口与后续验收

[NExT-QA](https://github.com/doc-doc/NExT-QA)、[STAR](https://github.com/csbobby/STAR_Benchmark)、[AGQA](https://cs.stanford.edu/people/ranjaykrishna/agqa/)、[Charades](https://prior.allenai.org/projects/charades)、[CLEVRER](https://clevrer.csail.mit.edu/)。EgoQA 候选视频来自 [VideoChat2 视频包](https://huggingface.co/datasets/ynhe/videochat2_data/blob/main/egoqa_split_videos.zip)，配套 [EgoQA 标注目录](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT/tree/main/video/vqa/ego_qa)可能需要数据集访问权限。来源相似不代表样本逐条等于作者抽样，必须保留实际归一化和媒体审计。

- [x] EgoQA 原始 7,813 条全部完成 32 帧解码审计，16 条缺失与 6 条无效媒体按固定清单排除，保留 7,791 条。
- [x] 五源统一制数输入检查通过：133,654 条有效 QA，媒体缺失 0，教师依赖检查通过。
- [x] 最终进度输出版本的训练/评测 smoke、真实 Agent 链路通过；错误、SIGTERM、SIGKILL watchdog 路径恢复原 8 个保活进程。
- [x] 第二次 full 故障定位到 OWLv2 超长文本输入；边界、来源迁移、训练池发布与首次 checkpoint 恢复的 69 项 CPU 回归通过，精确失败查询真实模型前向通过。
- [x] 本轮精确失败查询真实模型前向、正式 full 2,564→3,004 的无损推进，以及受控中断后原 8 worker 恢复均通过。
- [x] 本轮中断后同目录继续新增 71 条至 3,075；3,004 条完整终态前缀不变、重复执行 0，新增运行级错误 0。
- [x] 三个启用 CoT 的来源完成完整 Agent→转换→筛选，记录成功、拒绝、错误及合格数。
- [x] Instruct/AoTD 各训练完整 1 epoch，记录步数、loss、时间和 checkpoint；未宣称拥有完整训练资源峰值统计。
- [x] 作者/Base/Instruct/AoTD 同一完整划分评测，报告逐题及分组指标。

最终实测两轮训练共 17.17 小时，四模型并行评测约 1 小时；不含此前下载、制数和调试。详见结果报告。

本工程沿用 [AoTD](https://github.com/zhengrongz/AoTD)的研究任务；模型和数据分别遵循其发布方许可证，本仓库不重新分发这些大体积资源。
