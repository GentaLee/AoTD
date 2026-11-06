# 离线下载资源

本目录在 Git 中只保存说明与 [inventory.json](inventory.json)。模型权重、媒体、ZIP、wheel、环境和认证信息不提交到仓库。

| 内容 | 用途 |
|---|---|
| `wheels/` | 学生训练与评测的固定版本离线依赖 |
| `agent-wheels/` | 教师 Agent、解码、裁剪与流式 JSON 依赖 |
| `transfer-tools/` | 离线传输使用的 rsync 及动态库 |
| DeepSpeed 源码归档与来源记录 | 在兼容运行环境中构建离线 wheel |
| `llama_publisher_hashes.json` | 两份本地 Llama 与官方分片 SHA256 对照 |
| `egoqa_split_videos.zip` | EgoQA 原始视频归档；发布方哈希一致，配对及解码审计确认 16 条缺失、6 条无效媒体，详见固定排除清单 |

模型主要位于项目 `models/`；原始标注和媒体主要位于 `data/`。下载目录中的文件名存在，不表示资源已经完整或可用于训练。早期访问检查返回 403，也不代表后来通过本地导入验证的模型仍然缺失。

已有资源可关联到新 checkout：

```bash
bash use_local_assets.sh /absolute/path/to/existing/bundle --reuse-env
```

`--reuse-env` 仅用于同节点兼容环境。运行目录、环境及下载资源均需位于持久化挂载卷。异地部署使用 `sync_to_worker.sh`；主机、目标目录与站点参数由外部配置提供，不提交私人部署信息。

清单更新应在允许承担大量读盘/哈希的工作节点运行：

```bash
python -B tools/inventory_downloads.py \
  --downloads /absolute/path/to/downloads \
  --output downloads/inventory.json
```

EgoQA 原标注 7,813 条保留原样；[固定媒体排除清单](../reproduction/metadata/egoqa_media_exclusions.json)排除 22 条无法用于训练的记录，正式池使用其余 7,791 条。

清单记录文件名、大小、SHA256 和公开来源；认证目录、token、私钥及缓存不应出现在清单中。运行资源、验证结果与当前五源范围见[状态说明](../docs/REPRODUCTION_STATUS.md)。
