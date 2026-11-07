# AoTD 离线制数、训练与评测

本工程基于 [AoTD 原始仓库](https://github.com/zhengrongz/AoTD)与[论文](https://arxiv.org/abs/2412.01694)，提供离线教师 Agent 制数、CoT 转换与筛选、学生训练和 NExT-QA 评测入口。

**当前协议排除 ANetQA，使用 NExT-QA、STAR、AGQA、CLEVRER、EgoQA 五源，按固定媒体排除清单保留共 133,654 条 QA。EgoQA 原标注 7,813 条中，完整 32 帧解码审计确认 16 条媒体缺失、6 条无法解码；经明确授权按[固定排除清单](reproduction/metadata/egoqa_media_exclusions.json)排除这 22 条，保留 7,791 条，原始标注不改写。全部教师权重已齐备，最终进度输出版本的真实 Agent 小样本及四卡正式 batch 配置短训练、保存重载均通过；完整 CoT、正式一轮训练和完整评测尚未完成。** 原作者完整处理后数据和抽样 ID 未公开，本工程明确标记为五源工程重建，不宣称严格复现原论文训练池。

详见[当前状态、实测边界和资源说明](docs/REPRODUCTION_STATUS.md)、[验证证据](reproduction/evidence/verified_status.json)与[下载清单](downloads/README.md)。后续验证结果以证据文件的时间和字段为准。

## 一键运行

在已经配好离线资源的持久化挂载目录中执行：

```bash
bash run_aotd_full.sh
```

入口依次检查五源 QA 与媒体，按需运行真实 Agent→转换→筛选，独立训练 Instruct/AoTD 各 1 epoch，再对作者/Base/Instruct/AoTD 进行完整 NExT-QA val 评测。缺失标注、媒体、合格 CoT 或教师资源时会停止，不会用 smoke 数据代替。

```bash
bash run_aotd_full.sh --check             # 只检查输入，不启动 GPU 计算
bash run_aotd_full.sh --eval-only         # 只完整评测作者权重与 Base
bash run_aotd_smoke.sh                    # 实际短训练、评测与 checkpoint 重载
bash run_aotd_build_data.sh --check       # 检查五源制数输入
```

正式训练默认 4 张工作卡、4 张保活卡，ZeRO-2、BF16/SDPA、microbatch 1、梯度累积 12、有效 batch 48；单卡转换、筛选或评测阶段为 1 张工作卡、7 张保活卡。正常完成、错误和可处理信号退出后恢复原保活任务；已验证错误退出、SIGTERM 与 SIGKILL 后 watchdog 恢复路径，原 8 个保活 PID 恢复且无重复进程。默认每 30 秒显示进度，阶段完成立即输出；`AOTD_PROGRESS_INTERVAL_SECONDS` 支持 10–60 秒。

结果保存到 `results/full_*/`，包含日志、checkpoint、逐题 JSONL 和 `full_summary.json`。同一全量命令会复用最新受管理运行并自动续接制数、训练与评测；使用 `--new-run` 显式创建新的训练/评测目录。训练默认每 100 步保存完整 checkpoint，保留最近 2 个；`--resume RUN_DIR` 可指定受管理目录。运行中不写 Markdown 进度日志。

本次修复了生成程序 `UnboundLocalError` 被误判为运行级错误而中止全量制数的问题，保留已有记录并提供审核过的兼容迁移。故障原因、恢复规则和实测验证见 [开发记录](docs/DEVELOPMENT_LOG.md)。

## 离线资源与站点配置

Git 仅保存代码、配置、说明和审计元数据；模型、媒体、wheel、环境、缓存及凭证留在持久化挂载卷。

```bash
bash use_local_assets.sh /absolute/path/to/existing/bundle --reuse-env
```

仅在同节点、兼容环境中使用 `--reuse-env`；新环境通过 `setup_offline.sh` 与 `setup_agent_offline.sh` 安装本地 wheel。将站点专属变量写入忽略的 `runtime/site.env`，例如 `AOTD_KEEPALIVE_SCRIPT`。跨主机传输使用 `sync_to_worker.sh`，需要通过外部配置提供 `AOTD_WORKER_HOST`、`AOTD_WORKER_ROOT`；不要把部署主机、私人路径或凭证写进 Git。

模型与数据分别遵循其发布方许可证，下载入口和版本记录见详细说明及清单。源码目录直接位于仓库根目录，无需再套用旧 `src/AoTD` 补丁。
