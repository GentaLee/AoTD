# 第二轮数据与审核交付 / V2 data and audit delivery

按本次交付范围，代码、配置、文档及小型审计证据保留在 PR 中，完整数据和审核记录通过 [GitHub Release](https://github.com/GentaLee/AoTD/releases/tag/aotd-v2-data-audit-20261113) 提供。**不包含 AoTD 最终模型、Instruct 模型或两个训练 checkpoint。**

Code, configuration, documentation and compact evidence are in the PR. The full data/audit bundle is delivered through the linked release. **AoTD weights, Instruct weights and both training checkpoints are explicitly excluded.**

| 内容 / Contents | 包内位置 / Archive location |
|---|---|
| 133,654 QA、32,042 CoT 与 provenance | `data/five_source_sources/` |
| 实际训练输入、YAML、manifest / Training inputs | `data/five_source_train/` |
| 完整程序、转换、筛选、逐题与 worker 轨迹 / Full construction and worker traces | `data/construction/NExT-QA/`, `STAR/`, `AGQA/` |
| 四组完整评测、运行日志、状态与质量检查 / Evaluations, logs and checks | `results/full_*/` 顶层文件 / top-level files |
| 每个源文件与发布文件的大小、SHA256、脱敏标记 / Per-file size, hashes and sanitization | `DELIVERY_MANIFEST.json` |

## 下载与核验 / Download and verification

在持久化目录下载 Release 中的 `aotd-v2-data-audit-20261113.tar.gz` 与 `SHA256SUMS`，然后执行：

Download the archive and SHA256SUMS into persistent storage, then run:

```bash
sha256sum -c SHA256SUMS
mkdir -p aotd-v2-audit-extracted
tar -xzf aotd-v2-data-audit-20261113.tar.gz -C aotd-v2-audit-extracted
```

逐文件核验可在解压目录执行以下标准库脚本：

Verify every published payload file using Python's standard library:

```python
import hashlib, json
from pathlib import Path
root = Path("aotd-v2-audit-extracted/aotd-v2-data-audit")
manifest = json.loads((root / "DELIVERY_MANIFEST.json").read_text())
for item in manifest["files"]:
    path = root / item["path"]
    assert path.stat().st_size == item["bytes"], item["path"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == item["sha256"], item["path"]
print("PASS", len(manifest["files"]), "files")
```

核验示例要求 Python 3.11+。发布归档已逐文件重新读取校验，并从包内重算训练记录数、评测覆盖与正确数；证据见 [交付统计](../reproduction/evidence/artifact_delivery_20261113.json)。

The example requires Python 3.11+. The archive was read back and verified file by file; record counts, evaluation coverage and correct counts were independently recomputed from archive contents. See the delivery evidence linked above.

## 脱敏与复用边界 / Sanitization and reuse

- `/ARTIFACTS/v2` 表示本轮实验；`/ARTIFACTS/v1` 表示第一轮实验；`/ARTIFACTS/assets` 表示外部原始资源目录。其他私有路径为 `/EXTERNAL_PATH`。主机、IP 和设备型号替换为明确标记。
- 历史 manifest 中的哈希对应原始文件，不能误当成脱敏文件哈希。发布字节以 `DELIVERY_MANIFEST.json` 为准，原始 SHA256 仍保留作审计。
- 媒体、教师模型与环境未包含；按 [来源说明](REPRODUCTION_STATUS.md) 准备后映射路径，再在独立目录重新生成训练输入和校验，不覆盖旧实验，也不把历史恢复指纹当作可直接续跑状态。
- 归档不改变任何上游数据许可。完整响应与轨迹供审核，模型判定通过不等于人工确认每条推理都真实可靠。

Path placeholders distinguish the current experiment, prior experiment and external assets. Historical hashes document the original source; the delivery manifest hashes describe sanitized bytes. Provision external media/models, remap paths and regenerate training inputs/checks in a separate directory. This archive does not contain resumable model state, change upstream licenses or certify every teacher rationale as factually correct.

## 文件统计 / Package statistics

104,947 个数据文件，另含包内 README 和逐文件 manifest；压缩包 155,610,609 字节（148.40 MiB），发布数据文件合计 872,841,713 字节。49 个文件经过脱敏。QA 133,654 条，CoT 32,042 条，四组各 4,996 题，正确数复算一致。

The archive contains 104,947 payload files plus README and per-file manifest. Compressed size: 155,610,609 bytes (148.40 MiB); payload size: 872,841,713 bytes. 49 files were sanitized. Archive-derived QA/CoT counts and all four full evaluation totals match the reported results.

SHA256: `80bb1f9259266f49069b6439602b0bcefd897a420c6c8416c8c53dcbb1120d99`
