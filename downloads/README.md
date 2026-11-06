# 下载文件与持久化资源

此目录在 Git 中保存资源说明和 [inventory.json](inventory.json) 清单。模型、视频、ZIP、wheel、环境、缓存和认证凭证不提交到 Git。

当前 Shanghai 实际下载目录为 `/volume/yzhao04/workspace/aotd-nextqa/downloads`。Taihua 的运行资源位于 `/volume/pt-test/users/zhaoye/aotd-nextqa/`，运行不依赖 Shanghai 存储挂载。

| 资源 | 实际作用 | 状态判断 |
|---|---|---|
| `wheels/` | 学生训练/评测离线 Python 依赖，包括已构建的 DeepSpeed wheel | 按固定版本安装，目标机 `pip check` 验证 |
| `agent-wheels/` | Agent、视频裁剪和流式 JSON 处理的离线依赖 | 目标机导入和回归测试已通过 |
| `transfer-tools/` | 离线节点使用的 rsync 与动态库 | 保留在挂载卷，不安装到容器目录 |
| `deepspeed-0.15.4.tar.gz`、`deepspeed_source.json` | DeepSpeed 源码及来源元数据 | 构建放到 GPU Pod，禁用不需要的 CUDA 扩展 |
| `llama_publisher_hashes.json` | 官方/镜像 Llama 分片的发布方校验值 | 本地两份模型八个分片已逐个匹配 |
| `egoqa_split_videos.zip` | VideoChat2 EgoQA 候选视频包 | 以清单中的实测字节数和发布方 SHA256 为准；文件名存在不代表下载完成 |
| `teacher-access/` | 早期模型访问检查缓存 | 历史 403 不代表现在的本地教师缺失；当前以目标机模型验证记录为准 |

模型权重实际在 `models/`，原始训练标注、视频及 Charades 分块包主要在 `data/`，并非全部位于 `downloads/`。全部教师模型已齐备并在 Taihua 测试通过，详见 [当前状态](../docs/TAIHUA_STATUS.md)。

## 在新 checkout 复用已有下载

```bash
bash use_local_assets.sh /volume/yzhao04/workspace/aotd-nextqa --reuse-env
```

在 Taihua 对应使用：

```bash
bash use_local_assets.sh /volume/pt-test/users/zhaoye/aotd-nextqa --reuse-env
```

该脚本创建本地资源链接，不替换现有文件，也不把下载内容加入 Git。不同节点的环境不直接复用；如果 Python/CUDA 镜像不同，应运行 `setup_offline.sh` 和 `setup_agent_offline.sh` 重建环境。

## 更新清单

在同区域 GPU Pod 执行哈希扫描，避免 CPU 开发机承担大量读盘和哈希工作：

```bash
python -B tools/inventory_downloads.py \
  --downloads /volume/yzhao04/workspace/aotd-nextqa/downloads \
  --output downloads/inventory.json
```

清单只记录文件名、大小、校验值和公开来源，不包含 token、私钥或认证文件。
