# AoTD Taihua 离线复现：当前状态与运行说明

状态日期：2026-11-07（北京时间）。原始证据快照：2026-11-06 16:06 UTC。本文区分已经测得的结果、入口具备的功能和仍未完成的任务。

代码审查：[GentaLee/AoTD PR #1](https://github.com/GentaLee/AoTD/pull/1)。持续进度：[XFDG/AoTD issue #1](https://github.com/XFDG/AoTD/issues/1)；目标仓库关闭了 issues，因此跟踪项保存在本次 fork。代码导入不代表下列全量验收项完成。

**当前结论：全部教师模型和学生权重已落地 Taihua，教师前向、真实 Agent 制数小样本、四卡正式 batch 配置的短训练均通过。六源训练数据已有四源配齐；ANetQA、EgoQA 仍未完成。完整 CoT、全量学生训练与全量评测尚未完成，不能声称已复现论文指标。**

本分支将此前工作目录中的工程改动迁入 `GentaLee/AoTD`，直接使用仓库根目录的 `agent_system/`、`distillation/` 和 `evaluation/`。基础提交是 `1a7aa98f7ad5763b754a745d7d145541186c7bd7`；原工程参考提交为 `zhengrongz/AoTD@1b64726431ac3da212b42176165940339f3f556b`，两者上述源码目录相同。

## 1. 资源实际在哪里

| 位置 | 内容 |
|---|---|
| Shanghai：`/volume/yzhao04/workspace/aotd-nextqa/` | 原开发目录、下载与下载凭证之外的资源；保留原状 |
| Shanghai：`/volume/yzhao04/workspace/aotd-gentalee/` | 本次基于对应 GitHub 仓库创建的新分支 checkout |
| Shanghai：`/volume/yzhao04/workspace/aotd-nextqa/downloads/` | wheel、传输工具、发布方校验清单，以及下载中的 EgoQA 候选视频包 |
| Taihua：`/volume/pt-test/users/zhaoye/aotd-nextqa/` | 已配置运行环境、全部模型、本地训练/评测数据、脚本、缓存和结果 |
| Taihua：`/volume/pt-test/users/zhaoye/run_aotd_full.sh` | 原部署的一键全量入口包装脚本 |
| Taihua：`/volume/pt-test/users/zhaoye/aotd-pr-validation/` | 本分支根目录布局的独立验证 checkout，复用同节点资源，不替换原部署 |

Taihua 不需要挂载 `/volume/yzhao04`，也不在运行时访问 Hugging Face。模型、媒体、wheel、日志和环境均保留在挂载卷。Git 保存源代码、配置、说明和审计清单；**不把几十 GB 模型、视频、环境或 token 推进仓库**。[下载目录清单](../downloads/inventory.json)记录文件大小与 SHA256；下载过程中发生变化的文件单独标识。

在新 checkout 复用同一节点已有资源：

```bash
# Shanghai；仅做本地资源关联，重型验证应在配对 GPU Pod 执行
bash use_local_assets.sh /volume/yzhao04/workspace/aotd-nextqa --reuse-env

# Taihua，在新的 checkout 里执行
bash use_local_assets.sh /volume/pt-test/users/zhaoye/aotd-nextqa --reuse-env
```

`--reuse-env` 只适用于同节点兼容的 Python/CUDA 环境；新镜像应使用 `setup_offline.sh`、`setup_agent_offline.sh` 重建。跨节点同步由 Shanghai 执行 `sync_to_taihua.sh`：解引用资源链接，递归传输学生及教师分片，保留进度与断点；脚本需要当前环境配置好的 SSH 别名。首次迁移建议用 `AOTD_TAIHUA_ROOT` 指向新目录验证，只有默认目标才安装用户目录的三个包装入口。同步不会打包 HF 凭证检查目录；认证文件应一直留在独立的私有挂载目录。

## 2. 教师模型是否齐全

**齐全。原先 Hugging Face 的 403 已通过本地模型导入解决，不再是教师推理的阻塞项。** 两份 Llama 均为 BF16，各约 14.96 GiB，四个分片均与发布方 SHA256 匹配，传到 Taihua 后再次校验并实际加载。

下列路径相对于 Taihua 项目根目录：

| 角色 | 本地目录 | 来源及已验证内容 |
|---|---|---|
| 程序规划 | `models/teachers/DeepSeek-Coder-6.7B-Instruct` | [DeepSeek](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)，真实程序生成通过 |
| 视频问答专家 | `models/teachers/LLaVA-Video-7B-DPO` | [LLaVA DPO](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-DPO-hf)，视频前向通过 |
| 开放词汇目标检测 | `models/teachers/OWLv2` | [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)，检测前向通过；空检测结果也可能有效 |
| 时间定位 | `models/teachers/UniVTG` | [UniVTG 官方模型目录](https://drive.google.com/drive/folders/1-eGata6ZPV0A1BBsZpYyIooos9yjMx2f)，checkpoint、配置完整，实测输出区间 `[0,14]` |
| UniVTG 视觉编码器 | `models/teachers/CLIP` | [OpenAI ViT-B/32 权重](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)，发布方哈希一致，随 UniVTG 前向通过 |
| 答案选择 | `models/teachers/Meta-Llama-3-8B-Instruct` | [Meta Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)，本地 NousResearch 镜像权重与官方一致；BF16、有限 logits、答案选择通过 |
| 轨迹转换和解释筛选 | `models/teachers/Llama-3.1-8B-Instruct` | [Meta Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)，转换前向及正/负筛选对照通过 |

学生 Base [LLaVA-NeXT-Video-7B](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) 位于 `models/LLaVA-7B-base`，作者发布的 [AoTD-7B](https://huggingface.co/Zhengrongzz/AoTD-7B) 位于 `models/AoTD-7B-author`，均已齐备。公开 checkpoint 的 revision/分片哈希见[证据快照](../reproduction/evidence/verified_status.json)，两份 Llama 的发布方校验值另见[清单](../reproduction/metadata/llama_publisher_hashes.json)。

## 3. 六源数据现状

下表是 Taihua 已配对、归一化的数据，不能把“下载目录有同名文件”计作就绪。

| 训练源 | QA 条数 | 媒体 | 状态 |
|---|---:|---|---|
| NExT-QA train | 34,132 | 3,870 视频；缺失 0 | 就绪 |
| STAR train | 45,731 | 16,206 个唯一时间窗片段覆盖全部 QA；缺失 0 | 就绪 |
| AGQA train | 25,000 | 5,538 视频；缺失 0 | 就绪，确定性重建抽样 |
| CLEVRER train | 21,000 | 8,877 视频；缺失 0 | 就绪，确定性重建抽样 |
| ANetQA train | 目标约 25,000 | 标注和原视频尚未配对 | 缺失 |
| EgoQA train | 论文约 7,800，候选来源条数尚待核实 | 找到 VideoChat2 候选视频包；标注访问受限 | 缺失 |
| 已就绪合计 | **125,863** | 仅上述四源 | 六源尚未齐全 |

STAR/AGQA 使用的 Charades 官方全部 9,848 视频已落地；16 个分块归档共 16,339,546,533 字节，哈希/解压校验完成。STAR 按原生标注的 start/end 重新编码片段，16,206 个唯一时间窗处理完毕、错误 0。

NExT-QA train 有同一个 `video_qid=3617262562_2` 对应两道不同题的原生重复 ID（行 30235 与 32372），归一化保留两道题，后者增加行号后缀，并保留原始来源字段。CLEVRER 的多选题可能存在多个正确候选，因此重建保留原生逐候选 Yes/No 标签，不能强制改成单选。作者未发布完整六源抽样 ID、处理后训练 JSON、合格 CoT 或程序生成 few-shot 示例；本工程的确定性重建和提示词不等于作者原始数据序列化。

## 4. 缺失项从哪里下载

### ANetQA：有官方入口，原训练标注下载失效

- [官方项目页](https://milvlg.github.io/anetqa/) 和[官方 dataset 说明](https://github.com/MILVLG/anetqa-code/tree/main/dataset)。需要该项目的细粒度 ANetQA，不能用名称相似的 ActivityNet-QA 替代。
- [官方说明引用的 train QA 地址](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EbOk0tgkpZlIqwsW1yVq7PgB2jJUx0x0eCv5iu73hl--uQ?download=1)：实测原 SharePoint 域名 DNS 解析失败；这是失效的来源记录，**目前不是可用下载链接**。
- [ActivityNet 官方数据下载入口](https://activity-net.org/download.html)：原视频来源入口；本次尚未取得并完成 ANetQA 标注对应视频的完整覆盖验证。

仍需取得可访问的官方标注归档/经验证镜像，再按 ANetQA train 抽样并核对视频。当前没有已验证可下载的 ANetQA 标注镜像。

### EgoQA：候选视频来源已找到，配套标注仍需数据集访问权

- [VideoChat2 EgoQA 视频包页面](https://huggingface.co/datasets/ynhe/videochat2_data/blob/main/egoqa_split_videos.zip)。[固定版本直接下载](https://huggingface.co/datasets/ynhe/videochat2_data/resolve/eccfd0dc86fd0a03b5ca52bf65d0e9b11d7e9247/egoqa_split_videos.zip)。
- 已发现本地 `downloads/egoqa_split_videos.zip` 在持续增长，**尚未下载完成**。发布方字节数应为 `3680916665`，SHA256 应为 `4341cfed79cfbe96f93ebd6fc2dff6219abea8a6df8d9e2b9c0800624a885831`。完整性判断以[清单](../downloads/inventory.json)为准，不能提前解压或标记就绪。
- [配套标注目录](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT/tree/main/video/vqa/ego_qa)，包括 `train.json`、`instructions.json`；[train.json 页面](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT/blob/main/video/vqa/ego_qa/train.json)。固定版本为 `ccb392554b85d04df8e895f300870db73955abdd`。
- 当前匿名访问标注返回 401，已有 HF 凭证返回 403：需要在 **OpenGVLab/VideoChat2-IT 数据集页面**申请/同意访问条件；该权限与 Llama 模型访问权是两回事。不要把 token 发到 issue、PR 或聊天。
- [视频镜像及来源清单](https://huggingface.co/datasets/axiong/VideoChat2-egoqa-conversation)可核对上述视频包来源、固定版本、大小和哈希。其更大的 conversation 数据不自动等同于本次所需 EgoQA 标注。

视频包和标注取齐后，仍需检查原生 schema、问答条数、视频映射及与论文 EgoQA 的对应关系，然后写入 `data/raw/EgoQA/train_normalized.jsonl`。目前只确认找到候选来源，不声称已经补齐。

### 其他已经齐备的数据来源，供重新部署时使用

[NExT-QA](https://github.com/doc-doc/NExT-QA)、[STAR](https://github.com/csbobby/STAR_Benchmark)、[AGQA](https://cs.stanford.edu/people/ranjaykrishna/agqa/)、[Charades](https://prior.allenai.org/projects/charades)、[CLEVRER](https://clevrer.csail.mit.edu/)。训练源配置见 [sources.json](../config/construction/sources.json)。所有外网下载在 Shanghai 完成，再传 Taihua。

## 5. 已完成的验证与证据

目标硬件为 8×B200；实际环境：PyTorch `2.11.0a0+eb65b36914.nv26.02`、CUDA `13.1`、Transformers `4.49.0`、DeepSpeed `0.15.4`、NumPy `1.26.4`，使用 BF16/SDPA。未要求安装 Helion。GPU 任务前记录 `nvidia-smi` 和软件版本；实验日志保存在挂载卷。

| 验证 | 实测结果 | 边界 |
|---|---|---|
| 四个公开 Agent 组件 | DeepSeek、LLaVA、OWLv2、UniVTG 前向 PASS | 小样本功能测试 |
| 两个本地 Llama | BF16、有限 logits、答案选择、转换、正/负筛选对照 PASS | 一个简短转换 fixture 被筛选器拒绝，属于真实质量拒绝 |
| 真正的 Agent 程序执行→转换→筛选 | 4 道真实 train QA：3 条程序执行成功、1 条非代码回复被拒；1 条答对并完成转换、筛选、CoT 导出 | 只验证链路，不报告为全量准确率或合格率 |
| 四卡 ZeRO-2 短训练 | world size 4、microbatch 1、accumulation 12、有效 batch 48；2 步 loss `15.7559` / `15.8379`，梯度有限；checkpoint+processor 保存/重载成功 | 96 条受控重复 QA，仅用于配置验证，不是全量训练 |
| NExT-QA val 全量输入检查 | 4,996 问题 / 570 视频，媒体哈希、解码和 preflight 通过 | 没有全量推理分数 |
| 本分支布局回归 | Shanghai GPU 与 Taihua 独立目录均 10 项测试通过；Taihua CPU 解码/processor/collator、4,996 题 eval-only preflight 通过 | 迁入新布局后未重复完整 GPU 训练 |
| GPU 保活恢复 | 历史 GPU 验证结束后恢复原八卡保活；验证范围 OOM 0 | 不保证任意全量数据长度均不会 OOM |

可审查的原始字段与哈希：[verified_status.json](../reproduction/evidence/verified_status.json)；迁入本分支后的独立检查见 [repository_integration.json](../reproduction/evidence/repository_integration.json)。其中历史证据中的 `$AOTD_ROOT/src/AoTD/` 是原部署源码路径，不是新分支运行路径。Taihua 原始实验记录：

```text
results/validation_public_agent_20261106T115829/public_agent_validation.json
results/validation_batch48_fixed_20261106T110440/validation_summary.json
results/validation_local_llama_taihua_20261106T121838/llama_validation.json
results/validation_real_agent_20261106T122848/real_construction_validation.json
```

新 checkout 的轻量回归命令（在 GPU Pod 执行，使用 CPU 即可）：

```bash
source runtime_env.sh
CUDA_VISIBLE_DEVICES='' env/bin/python -B tools/test_construction.py -v
CUDA_VISIBLE_DEVICES='' env/bin/python -B tools/cpu_validate.py
bash run_aotd_full.sh --eval-only --check
```

运行环境/下载资源不会因 Git clone 自动具备；以上命令依赖已经部署或通过离线安装配好的资源。

## 6. full 脚本到底做什么

在 Taihua 原部署的一键命令：

```bash
bash /volume/pt-test/users/zhaoye/run_aotd_full.sh
```

在本分支的新 checkout，使用 checkout 自己的入口：

```bash
bash run_aotd_full.sh
```

完整流程依次为：

1. 如果正式训练输入不存在，调用 `run_aotd_build_data.sh`，先检查六源标注、媒体和教师资源，再归一化 QA，运行 Agent 程序、真实视觉工具、正确答案门控、Llama-3.1 转换及解释筛选，导出 QA/CoT。
2. 准备 `instruct.yaml`、`aotd.yaml` 并检查训练记录和全部评测媒体。**目前六源未齐，默认 full 会明确报缺失并停止，不能跳过数据缺口得到虚假的“全量完成”。**
3. Instruct（QA-only）和 AoTD（QA+CoT）各自从相同 Base 独立初始化，分别训练完整 1 epoch；默认 4 卡、microbatch 1、累积 12、有效 batch 48、ZeRO-2、BF16/SDPA、seed 42，没有 smoke 的步数/样本截断。
4. 对作者权重、Base、自己训练的 Instruct、AoTD 分别评测完整 NExT-QA val 4,996 题；`AOTD_EVAL_SPLIT=test` 可改为 8,564 题，需要对应媒体齐备。不能把两种划分混报。
5. 保存训练 checkpoint、processor、逐题输出、统计及 `full_summary.json`。只有每组覆盖完整评测划分才会写最终 PASS。

只检查资源、不启动 GPU：`bash run_aotd_full.sh --check`。只做作者/Base 完整评测：`bash run_aotd_full.sh --eval-only`。`--eval-only` 不会训练自己的 Instruct/AoTD，不能当作完整复现。

制数可独立检查 `bash run_aotd_build_data.sh --check`，或在已齐备源上先运行 `bash run_aotd_build_data.sh --dataset NExT-QA`。默认 full 仍要求六源。制数记录为可恢复 JSONL，按输入/提示词/代码/模型指纹校验；重跑同一制数阶段会续接记录。**学生训练入口不承诺自动从任意中断 checkpoint 续训**，重新 full 会创建新运行目录。

运行时显示阶段、样本进度、估计剩余时间、训练 step、评测进度，并定期输出 GPU 利用率。原保活进程树通过 `SIGSTOP/SIGCONT` 暂停/恢复，watchdog 处理异常退出；训练/程序构建默认 4 张工作卡+4 张保活卡，单卡转换/筛选/评测时为 1 张工作卡+7 张保活卡。保活不是有效训练计算，不能把四卡训练写成八卡训练。原保活脚本可通过 `AOTD_KEEPALIVE_SCRIPT` 指定。

## 7. 结果、日志与关键修复

完整运行产物位于 `results/full_<UTC时间>_<PID>/`：`run.log`、两组训练 checkpoint、各组评测 JSON/逐题 JSONL、`full_summary.json`。制数运行日志位于 `results/build_<UTC时间>_<PID>/`，持久轨迹/进度保存在 `data/construction/`，正式训练 QA/CoT 与 YAML 位于 `data/original_sources/`、`data/original_six_source/`。终端会打印本次确切路径。运行过程不写 Markdown 进度文件；本文只是人工审查的状态说明。

工程改动中影响结果的部分：

- 修正 Trainer 对 `**kwargs` 的错误 loss 能力推断，设置 `model.accepts_loss_kwargs=False`。原模型输出 microbatch mean CE，防止累积 12 时 loss/梯度被错误放大 12 倍；保留原版 padding-only label collator 的监督范围，未暗中切换成 answer-only loss。
- 显式连接 DeepSpeed 配置、优化器、设备与保存路径，并输出真实运行引擎/world size/ZeRO stage，避免仅配置文件看似正确。
- 修复教师初始化、显式设备映射、时间窗与标量/列表观察处理；视频问答错误不再随机选答案。
- 程序生成使用允许的工具/AST 检查和超时；非代码回复、缺乏真实视觉轨迹、答错或解释筛选失败均保留拒绝记录。它是执行约束，不是隔离任意不可信代码的安全沙箱。
- JSONL 尾部中断恢复、逐条落盘、来源/提示词/代码/模型指纹和有界索引，避免长期任务累积内存以及混用过期轨迹。

## 8. 未完成项与验收条件

- [ ] 取得可访问的 ANetQA train 标注与配套原视频，完成归一化及媒体覆盖审计。
- [ ] 完成 EgoQA 候选 ZIP 下载和发布方哈希验证，取得受限标注，核对来源/条数/视频映射，完成归一化。
- [ ] 六源统一 preflight 全部通过；取得作者处理后数据则单独记录来源，否则继续明确为工程重建。
- [ ] 在需 CoT 的来源上跑完整 Agent→转换→筛选，记录成功/拒绝/错误数及最终合格 CoT 数，修复运行中的真实 bug。
- [ ] 正式训练 Instruct/AoTD 各 1 epoch，保存日志、实际步数、loss、资源峰值与 checkpoint。
- [ ] 作者/Base/Instruct/AoTD 在同一完整评测划分产生逐题输出和分组指标，再报告 CoT 增益。

完整耗时尚未实测。两步短训练、四题 Agent 的速度不足以外推六源完整工作量；实际耗时取决于工具调用次数、视频长度、合格 CoT 数以及两组训练记录数。正式报告应在长阶段运行后依据实际吞吐估计，而不是把 smoke 的时长当作全量承诺。
