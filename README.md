# AoTD 原版复现与 Taihua 离线运行

> **当前进展（2026-11-07，北京时间）：** 教师模型齐全并通过 Taihua 实际前向；真实 Agent 小样本链路和四卡 batch 48 短训练通过。六源数据已有四源、125,863 条 QA 配齐，ANetQA 与 EgoQA 尚未就绪，全量制数、训练和评测尚未完成。
>
> 请先阅读 [当前状态、缺失资源下载链接与完整运行说明](docs/TAIHUA_STATUS.md)。[下载文件清单](downloads/README.md)和[实测证据](reproduction/evidence/verified_status.json)随分支保存；权重、媒体、环境、日志和凭证不提交 Git。
>
> 本分支已经集成工程修复，源码位于仓库根目录。配置好本地离线资源后执行 `bash run_aotd_full.sh`；`bash run_aotd_full.sh --check` 只检查输入。完整入口会先检查六源，当前缺失数据会使其停止，不能把入口已实现理解为全量已跑完。
>
> 以下保留原版复现方法参考，其中从零 clone、旧环境及 guide/KIT 补丁步骤描述的是原始工作流；使用本分支无需再向 `src/AoTD` 套补丁。部署、实际测试结果与当前阻塞均以新版状态说明为准。

# 原版复现方法参考

仅使用 NExT-QA 作为评测数据集。修订日期 2026年11月6日。

本指南用于复现 AoTD 原版：使用多模型 Agent 构建程序执行轨迹，经转换与筛选得到 CoT，再对 LLaVA-NeXT-Video-7B 学生进行答案与解释双任务监督训练。内容覆盖作者权重验证、原版训练数据、Agent 制数、学生训练和 NExT-QA 评测。

评测仅使用 NExT-QA。原版训练仍保留论文的六源训练池；只保留一个评测数据集，并不意味着将训练数据也缩减为 NExT-QA。严格训练复现所需的完整作者训练数据未随本指南提供，具体获取与重建边界见第八节。

原指南撰写时只完成 CPU 检查；此后已在目标 GPU 完成教师前向和四卡短训练，完整评测仍未执行。最新证据与边界请参见上方状态说明。

## 一 先固定评测划分

NExT-QA 是唯一评测数据集；它内部仍有 train、val、test 三个划分。官方 AoTD 示例默认读取 val.csv。本指南默认沿用 val.csv，方便与仓库设置对照，并提供官方 test.csv 的切换方式。不能把 val 与 test 的结果合并，不能用 test 调参或制数。

本次实际核对 NExT-QA 固定提交 `2432e9724f88ed9f40010e2989f104570a91de4e` 的标注：

| 划分 | 问题数 | 视频数 | 用途 |
|---|---:|---:|---|
| train.csv | 34132 | 3870 | 训练与教师制数 |
| val.csv | 4996 | 570 | 仓库默认评测划分，也可用作开发验证 |
| test.csv | 8564 | 1000 | 可选的官方最终测试划分 |

如果你明确要官方 test，请在后文设置 EVAL_SPLIT=test。若以 val 为论文结果，报告写清 NExT-QA val，避免让读者误认为官方 test。

## 二 原版复现配置

| 项目 | 配置 |
|---|---|
| 教师系统 | DeepSeek 规划、UniVTG、OWLv2、LLaVA 视频专家与 Llama |
| 学生 | LLaVA-NeXT-Video-7B |
| 制数流程 | 工具程序执行轨迹、CoT 转换与筛选 |
| 训练池 | 论文原六源 QA 与合格 CoT |
| 训练方式 | 答案与解释记录混合交叉熵训练 |
| 评测数据集 | 仅 NExT-QA，固定 val 或 test 划分 |

至少比较 Base、QA-only Instruct、QA+CoT AoTD 三个学生状态。自己的 Instruct 与 AoTD 从相同 Base 权重独立初始化；判断 CoT 的增益应比较 Instruct 与 AoTD。作者发布权重的评测结果单列，避免与自行训练结果混淆。AoTD 属于离线 CoT 蒸馏与监督微调，不是 OPD。

## 三 目录和软件环境

以下命令在 Linux NVIDIA GPU 服务器的 Bash 中执行。将本指南及 tools 目录上传到 `/data/aotd_nextqa/guide`。本机 Windows 上不直接运行 CUDA 训练。

```bash
export ROOT=/data/aotd_nextqa
export KIT="$ROOT/guide"
export REPO="$ROOT/AoTD"
export DATA="$ROOT/data/nextqa"
export VIDEO="$DATA/NExTVideo"
mkdir -p "$ROOT/models" "$ROOT/logs" "$ROOT/results" "$DATA"
git clone https://github.com/zhengrongz/AoTD.git "$REPO"
git -C "$REPO" checkout 1b64726431ac3da212b42176165940339f3f556b
```

仓库历史环境是 Python 3.10、PyTorch 2.1.2 cu121、FlashAttention 2.6.3。下面给出固定依赖起点；其中部分版本由本指南补充，完整 CUDA 镜像未在本地验证。

```bash
conda create -n aotd-nextqa python=3.10 -y
conda activate aotd-nextqa
python -m pip install --upgrade pip wheel packaging ninja
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.23.5 transformers==4.49.0 \
  accelerate==1.3.0 datasets==2.21.0 huggingface_hub==0.28.1 \
  decord==0.6.0 Pillow imageio opencv-python==4.11.0.86 \
  PyYAML==6.0 sentencepiece==0.2.1 protobuf tqdm
python -m pip install deepspeed==0.15.4 bitsandbytes==0.43.3
python -m pip check
```

原版的默认训练注意力需要另外安装 `flash-attn==2.6.3 --no-build-isolation`，或在 CLI 显式选择 sdpa 并记录差异。

若使用 B200 等 Blackwell GPU，不执行上述旧 cu121 组合，先采用支持该 GPU 的 PyTorch CUDA 镜像，再固定 Transformers 等依赖并做模型级 smoke。PyTorch 2.7 发布说明介绍了 Blackwell 与 cu128 支持，但不代表本指南验证了该组合下的 AoTD 训练。参考 https://pytorch.org/blog/pytorch-2-7/ 。准备 ffmpeg、unzip 及镜像所需图形解码动态库，FlashAttention 编译还需要匹配的 toolkit 与编译器。

```bash
nvidia-smi > "$ROOT/logs/nvidia-smi.txt"
python -m pip freeze > "$ROOT/logs/pip-freeze.txt"
python - <<'PY'
import torch, transformers
print(torch.__version__, torch.version.cuda, transformers.__version__)
assert torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
x = torch.ones((64,64), device='cuda', dtype=torch.bfloat16)
print((x @ x).sum().item())
PY
```

模型推理显存和训练显存需用 1 条样本、micro-batch 1 实测。Agent 制数与学生训练可分阶段运行，不要求同时驻留。本文用四卡举例，不承诺某型号 GPU 一定能装下。

## 四 准备唯一评测数据集

```bash
git clone https://github.com/doc-doc/NExT-QA.git "$ROOT/NExT-QA-source"
git -C "$ROOT/NExT-QA-source" checkout \
  2432e9724f88ed9f40010e2989f104570a91de4e
cp "$ROOT/NExT-QA-source/dataset/nextqa/"{train,val,test}.csv "$DATA/"
export EVAL_SPLIT=val
export EVAL_CSV="$DATA/$EVAL_SPLIT.csv"
```

原视频从 NExT-QA 官方 README 的 Raw videos 链接下载：https://github.com/doc-doc/NExT-QA 。需要 RGB 原视频，不是预提取特征。假设解压到 `/data/nextqa_raw`，为三个划分建立同一个 ID.mp4 符号链接视图：

```bash
for split in train val test; do
  python "$KIT/tools/prepare_data.py" link-nextqa \
    --csv "$DATA/$split.csv" --raw-root /data/nextqa_raw \
    --video-dir "$VIDEO" --report "$ROOT/logs/links_$split.json"
  python "$KIT/tools/prepare_data.py" audit-nextqa \
    --csv "$DATA/$split.csv" --video-dir "$VIDEO" \
    --report "$ROOT/logs/audit_$split.json" --decode
done
```

工具检查标签 0至4、文件存在性、重复 ID 冲突及媒体首尾帧解码。解码检查不代表所有中间帧完好；正式推理遇到坏媒体会停止，不补零帧。若本轮只做 val，可先准备 train 与 val，官方 test 评测前再补齐 test 视频。不要把缺失样本静默删除后仍报告完整数据集。

切换为官方 test 时，只需重新设置两个变量，后续所有评测命令均使用它们：

```bash
export EVAL_SPLIT=test
export EVAL_CSV="$DATA/$EVAL_SPLIT.csv"
```

测试划分一旦用于最终报告，应先在 val 上确定参数，再冻结流程做 test。评测脚本校验 CSV 文件名与 split 标签一致。

## 五 应用原版训练与评测工程修订

```bash
python "$KIT/tools/patch_official.py" --repo "$REPO" \
  --with-agent --dry-run
python "$KIT/tools/patch_official.py" --repo "$REPO" --with-agent
git -C "$REPO" diff > "$ROOT/logs/original_engineering.patch"
cp "$REPO/repro_patch_manifest.json" "$ROOT/logs/"
```

仅支持固定的干净 AoTD 版本，拒绝重复修改。本版只修订 NExT-QA 评测入口，不要求运行其他 benchmark。

修订包括训练 CLI 和 DeepSpeed 接线、保存 processor、Agent 写入 program output、NExT 数据返回结构、训练模板的评测设置冲突，以及确定性答案解析和逐样本结果保存。原 collator 的 padding-only labels 保留。原优化器默认是 adamw_bnb_8bit；本文 ZeRO-2 主命令显式选 adamw_torch，作为需记录的工程调整。

原始四卡模板的有效 batch 是 4乘4乘1等于16，论文写48。本文采用四卡 micro-batch 1、累积12，使有效 batch 为48。原代码传入的 --deepspeed 未被 parse_args 读取，补丁修复这一接口。

## 六 下载原版模型

```bash
huggingface-cli download Zhengrongzz/AoTD-7B \
  --revision f0e74be6caf57cf84d5819919abefd80a826d1fb \
  --local-dir "$ROOT/models/AoTD-7B-author"
huggingface-cli download llava-hf/LLaVA-NeXT-Video-7B-hf \
  --revision f417d9edeb69a9eaa88283780332d00ce3c66593 \
  --local-dir "$ROOT/models/LLaVA-7B-base"
```

作者权重用于评测验证，自己的 Instruct 和 AoTD 都从 LLaVA-7B-base 独立初始化。它的语言骨干为 Vicuna-7B-v1.5。保留全部 tokenizer、processor、config 和权重分片，不把 DPO 视觉专家误当作学生初始权重。

## 七 先验证作者权重

```bash
cd "$REPO/evaluation"
CUDA_VISIBLE_DEVICES=0 python eval_nextqa.py \
  --model_path "$ROOT/models/AoTD-7B-author" \
  --data_dir "$EVAL_CSV" --video_dir "$VIDEO" \
  --split "$EVAL_SPLIT" --limit 2 \
  --save_dir "$ROOT/results/author_${EVAL_SPLIT}_smoke.json"
```

查看 raw_output 是否正确解析为选项、题目和标签是否对应。确认后去掉 --limit，使用新的 full 输出文件名完成全量。此处是作者模型的评测复现，不等于已经自行训练。

论文 NExT-QA 参考分数为 Base 61.6、Instruct 77.1、AoTD 77.6。只有相同划分与协议才适合对照；不能把这些数值当作官方 test.csv 的达标阈值。

## 八 准备原版训练数据

原文训练数据并非只有 NExT-QA：

| 来源 | QA数量 | 合格CoT数量 |
|---|---:|---:|
| AGQA | 25.0K | 5.4K |
| ANetQA | 25.0K | 3.6K |
| STAR | 45.7K | 11.2K |
| NExT-QA | 34.1K | 12.1K |
| CLEVRER | 21.0K | 无 |
| EgoQA | 7.8K | 无 |
| 合计 | 158.6K | 32.3K |

严格训练复现需要取得相应 QA 抽样、媒体、合格 CoT 和提示词。当前仓库没有给出可直接下载并加载的完整六源 CoT 清单，YAML 仍是路径占位，不能编造一个下载地址或用几条示例替代。

如果已有作者处理数据，按样本 ID 与原始视频配对并建立原版训练 YAML。如果需要重建，使用原 Agent；仓库数据读取入口直接支持 STAR 和 NExT-QA，其他来源须实现相应适配，或取得作者处理好的数据。因为只测 NExT-QA，不需要下载其他评测 benchmark，但原版六源训练媒体仍然需要。

### 原Agent的组件

| 环节 | 模型 |
|---|---|
| 程序规划 | DeepSeek-Coder-6.7B-Instruct |
| 时间定位 | UniVTG checkpoint |
| 目标检测 | google/owlv2-base-patch16-ensemble |
| 视频问答与动作识别 | llava-hf/LLaVA-NeXT-Video-7B-DPO-hf |
| Agent答案选择 | Meta-Llama-3-8B-Instruct 默认配置 |
| CoT转换与筛选 | Meta-Llama-3.1-8B-Instruct |

在独立 Agent 环境补齐 omegaconf、rich、ftfy、einops、timm、h5py、scipy、ffmpeg-python、tensorboardX 等依赖，并配置 UniVTG 权重。官方 requirements 中 ffmpeg=python==0.2.0 是无效写法，应为 ffmpeg-python==0.2.0。Llama 访问权限由账号自行完成。每个组件与 prompt 保存 revision 或哈希。

### 使用NExT训练划分检查Agent

修改 `agent_system/configs/base_config.yaml`，填各组件绝对模型路径，并设置：

```yaml
cot:
  dataset_name: NEXT_QA
  dataset_dir: /data/aotd_nextqa/data/nextqa/train.csv
  video_dir: /data/aotd_nextqa/data/nextqa/NExTVideo
  save_cot_dir: cot/raw_next_smoke/part000.txt
```

```bash
cd "$REPO/agent_system"
mkdir -p cot/raw_next_smoke
CUDA_VISIBLE_DEVICES=0 python cot_construction.py --limit 1
python cot_transfer.py --model_path /data/models/Llama-3.1-8B \
  --cot_path cot/raw_next_smoke --save_dir cot/next_transfer_smoke.json
python cot_filter.py --model_path /data/models/Llama-3.1-8B \
  --cot_path cot/next_transfer_smoke.json --save_dir cot/next_filter_smoke.json
```

转换器的 cot_path 是目录，目录只放轨迹文本。记录要包含 question id、video id、程序、执行轨迹、answer 与 program output。检查程序与视觉结果，再扩大到训练集；全量使用新的目录与输出名，不能追加 smoke 造成重复。

程序与转换 prompt 中仍有示例占位，应从论文附录核对并补齐所用示例。正确答案筛选及文本自审均不能保证全部视觉事实正确，需要抽查。

### 整理学生训练记录

统一中间格式为 sample_id、dataset、video_id、绝对 video_path、question、options、answer。混合来源的 ID 要同步命名并在 QA 与 CoT 中保持一致。答案记录与解释记录使用不同提示词，解释只放 assistant 目标。

本指南的工具可以将已经合并、核对好的六源 QA 和 CoT 转成官方训练格式：

```bash
python "$KIT/tools/prepare_data.py" prepare-train \
  --qa /data/original_sources/qa_normalized.json \
  --cot /data/original_sources/cot_filtered.json \
  --output-dir "$ROOT/data/original_six_source"
export TRAIN_DATA="$ROOT/data/original_six_source"
```

这两个源文件是需实际准备的输入，不是随包附带的数据。工具输出 answering.json、reasoning.json、instruct.yaml、aotd.yaml 和 manifest.json，检查 ID、问题、视频、答案和解释非空，不替你证明数据来自作者。

NExT-QA 样本 ID 使用 videoID_qid，并在原始 QA、Agent 轨迹、CoT 与训练记录中保持一致。

## 九 训练学生

```bash
cd "$REPO/distillation"
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus 4 train.py \
  --model_path "$ROOT/models/LLaVA-7B-base" \
  --data_yaml "$TRAIN_DATA/aotd.yaml" \
  --output_dir "$ROOT/models/train_smoke" \
  --deepspeed configs/ds_config_zero2.json --optim adamw_torch \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --max_steps 2 --seed 42
```

检查有限 loss、ZeRO-2 日志、视觉 token 与特征匹配、checkpoint 和 processor。用 smoke checkpoint 再评测 1条，分数不入正式结果。

```bash
cd "$REPO/distillation"
for mode in instruct aotd; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus 4 train.py \
    --model_path "$ROOT/models/LLaVA-7B-base" \
    --data_yaml "$TRAIN_DATA/$mode.yaml" \
    --output_dir "$ROOT/models/${mode}_s42" \
    --deepspeed configs/ds_config_zero2.json --optim adamw_torch \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 12 \
    --seed 42
done
```

正式训练不传 max_steps 2。设置为 1 epoch、lr 4e-5、cosine、warmup 0.03、BF16、32帧、序列上限8192；冻结视觉编码器，训练投影层与语言模型。QA与CoT混合数据的1 epoch步数不同，保存实际 global_step、监督量和成本。此模板并非显式分支归一化的双损失实现。

## 十 仅在NExT QA评测

```bash
cd "$REPO/evaluation"
for tag in author base instruct aotd; do
  case "$tag" in
    author) checkpoint="$ROOT/models/AoTD-7B-author" ;;
    base) checkpoint="$ROOT/models/LLaVA-7B-base" ;;
    *) checkpoint="$ROOT/models/${tag}_s42" ;;
  esac
  CUDA_VISIBLE_DEVICES=0 python eval_nextqa.py \
    --model_path "$checkpoint" --data_dir "$EVAL_CSV" \
    --video_dir "$VIDEO" --split "$EVAL_SPLIT" \
    --save_dir "$ROOT/results/${tag}_${EVAL_SPLIT}_full.json"
done
```

32帧，greedy，最大新生成1024 tokens，要求直接输出选项字母。没有传 limit 才进行全量。若曾用同名输出，换新文件名；脚本拒绝覆盖。

## 十一 指标和结果

评测使用 strict_letter_v1 确定性解析：接受明确字母格式，无法解析则计错并统计invalid。移除原脚本的随机猜选项；这会与旧解析协议产生差异，应在复现报告声明。

每次评测保存JSONL逐题预测和完整结束后的JSON汇总，包含原始输出、预测/真实索引、题目、视频、qid、类别、split、CSV哈希、脚本哈希、模型路径和是否完整。

| 指标 | 报告方式 |
|---|---|
| Overall Accuracy | 正确题数除以全部评测题数 |
| C类准确率 | 因果类细分类按题数加权 |
| T类准确率 | 时序类细分类按题数加权 |
| D类准确率 | 描述类细分类按题数加权 |
| Invalid Rate | 无法明确解析的输出占全部题数比例 |
| 资源成本 | 制数、训练、推理分开记录 |

不要用细分类准确率的简单平均代替overall；不要剔除invalid或缺媒体后改变分母。中断时只有部分JSONL，不能当作全量结果；本脚本未实现隐式断点续跑，修复后用新名字完整重跑。

最终分别填表，val和test分列：

| 版本 | 训练池 | NExT划分 | N | Acc | C | T | D | Invalid |
|---|---|---|---|---|---|---|---|---|
| 作者权重 | 作者训练 | 明确填写 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| Base | 无此次训练 | 同上 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| Instruct | 原六源 QA | 同上 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| AoTD | 同一QA加Agent CoT | 同上 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |

先做seed42全链路，再按预算增加43、44；报告单次或mean加减std。全量JSON应满足 meta.full_dataset=true 且 summary.total等于meta.dataset_size。

## 十二 常见故障和验收

| 现象 | 检查 |
|---|---|
| 解析失败多 | raw_output、chat template、选项顺序与processor，不随机填答案 |
| 找不到ID.mp4 | 原视频与特征混淆、多级目录、链接或ID冲突 |
| ZeRO没有启用 | 原版CLI补丁、--deepspeed参数和真实训练日志 |
| OOM | micro-batch1、ZeRO、梯度检查点、序列与像素预算；改动独立记录 |
| Agent转换为空 | program output字段、正确答案格式、输入目录与prompt示例 |
| 分数不同于论文 | 划分、模型初始化、教师制数、训练池、监督方式与解析协议 |

本包的 5 项 CPU 检查涵盖答案解析、C/T/D 加权、NExT-QA 样本 ID、原版补丁和结果完整性；检查未执行真实 GPU 模型前向。

```bash
python "$KIT/tools/verify_nextqa.py" --repo "$REPO" \
  --scratch "$ROOT/checks_fresh"
```

验收产物包括独立训练的学生权重、processor、训练与教师数据manifest、完整逐样本预测、分类指标、环境版本、代码diff、原始/修改配置和失败统计。只运行作者权重属于评测复现；自行训练后再评测才构成训练复现。本指南不要求额外评测数据集。

## 十三 一手来源

- AoTD代码与论文 https://github.com/zhengrongz/AoTD 和 https://arxiv.org/abs/2412.01694
- NExT-QA标注与原视频 https://github.com/doc-doc/NExT-QA
- AoTD发布权重 https://huggingface.co/Zhengrongzz/AoTD-7B
