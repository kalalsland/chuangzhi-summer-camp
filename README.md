# 创智夏令营 · Harness Engineering

> **SII 2026 夏令营 · Harness Engineering 考核提交方案**
> 在 **2048 token 预算** + **Qwen3-8B 冻结权重** + **仅 stdlib/numpy** 的硬约束下，构建一个检索式 few-shot Harness，在客服意图分类 / OOD 分类 / A/B/C/D 选择题三类混合任务上取得高准确率。

📄 [**探索报告.pdf**](探索报告.pdf) · [**探索报告.md**](探索报告.md) · [English](README.en.md)

---

## ✨ 最终结果（DashScope 部署，4 轮平均）

| 指标              | 值         | 备注                                    |
| ----------------- | ---------- | --------------------------------------- |
| **DEV 准确率**    | **85.2%**  | 官方测试集，4 轮 85.3/85.3/85.0/85.0    |
| OOD mock          | 82.0%      | 自造 52 类跨域数据集                    |
| MCQ mock          | 87.2%      | 自造 A/B/C/D 均衡选择题                 |
| **估计加权**      | **83.7%**  | 官方权重 DEV 20% / OOD 60% / MCQ 20%    |
| Prompt 平均用量   | 1426 token | 远未撞 2048 预算                        |
| 抗注入 mock       | **9 / 9**  | 9 种注入向量全挡                        |

---

## 🧠 方案概览

一个 **检索式 few-shot Harness**，在 `update(text, label)` 阶段建立外部记忆与索引，在 `predict(text)` 阶段按任务类型组装 prompt，核心要点：

- **检索**：**RRF 融合**（余弦 + BM25）从外部记忆中召回 top-k 邻居
- **Prompt 组装**：few-shot 示例 + LLM 生成的 label 语义描述 + 抗注入封装
- **任务路由**：训练文本启发式判别 MCQ vs 分类，走不同的 system message 与 reasoning 模式
- **鲁棒性**：LLM 异常时回退到 top-1 检索标签；MCQ 选项 stem 扫描，避免脆弱的启发式

完整演化历程（P1 基线 → 8 次失败 → 4 次反转 → 2 项加固）与实验数据见 [探索报告.md](探索报告.md)。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖：`openai`、`transformers`、`numpy`。推荐 Python 3.10+。

### 2. 配置 LLM API Key

编辑 [llm_client.py](llm_client.py)，填入你的 API Key：

```python
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY  = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"   # 替换为你自己的
MODEL    = "qwen3-8b"
```

> 默认走阿里云 DashScope（Qwen3-8B 非思考模式）。也可切换到硅基流动——见文件内注释。

### 3. 验证连通性（可选）

```bash
python llm_client.py
```

输出 `✓ Connected. Response: ...` 即表示 API Key / 网络 / 模型均可用。

---

## 🧪 运行说明

### 基础评测

```bash
python run.py                          # 默认 4 轮评测，workers=20
python run.py --workers 100            # 增加并发（受 API QPS 限制）
python run.py --runs 1                 # 只跑 1 轮（快速调试）
```

**默认参数**（见 [run.py](run.py)）：

| 参数                    | 默认值                     | 含义                                   |
| ----------------------- | -------------------------- | -------------------------------------- |
| `--train`               | `data/train_dev.jsonl`     | 训练流 JSONL（`{text, label}`）        |
| `--dev`                 | `data/test_dev.jsonl`      | 验证集 JSONL                           |
| `--workers`             | `20`                       | 并发线程数，直接影响吞吐与 API 稳定性  |
| `--max-prompt-tokens`   | `2048`                     | 单 prompt 截断阈值，与官方评分系统一致 |
| `--runs`                | `4`                        | 评测轮数，最终报平均 ± 各轮列表        |

**样例输出**：

```text
============================================================
  本地调试评测
============================================================
  Train: 231 条 | Dev: 539 条
  max_prompt_tokens: 2048 | runs: 4

  [Run 1/4]
    进度: 539/539
    准确率=85.3%  耗时=68.2s
  ...
============================================================
  平均准确率: 85.2%  (各轮: 85.3%, 85.3%, 85.0%, 85.0%)
  prompt/条:  1697 token
  compl/条:   25.8 token
  总耗时:     281.4s
```

### 一键测试（带日志落盘）

```bash
./run_test.sh                          # 默认参数
./run_test.sh --runs 1 --workers 50    # 透传参数给 run.py
```

等价于调用 [test/run_with_log.py](test/run_with_log.py)，会在 `output/<时间戳>/` 下保存：

- `run.log`：完整 stdout/stderr
- `meta.json`：命令行参数、起止时间、退出码、主机信息
- `solution.py`：当次运行时 `solution.py` 的快照，便于 diff 不同实验

> Windows 直接用 Git Bash 运行即可；也可以手动 `python test/run_with_log.py` 调用。

### 换数据集跑 OOD / MCQ

仓库已附 3 类训练+测试数据（[data/](data/)）：

```bash
# OOD 跨域分类 mock
python run.py --train data/train_ood.jsonl --dev data/test_ood.jsonl

# A/B/C/D 选择题 mock
python run.py --train data/train_mcq.jsonl --dev data/test_mcq.jsonl
```

### 诊断脚本（[test/](test/)）

| 脚本                              | 用途                                                                |
| --------------------------------- | ------------------------------------------------------------------- |
| `test/api_determinism_probe.py`   | 5 次同 prompt 判断 API 是否 deterministic（决定自一致性投票有效性） |
| `test/hard_case_analysis.py`      | 跑完整 dev，拆解"检索 miss"vs"LLM 决策错"，输出混淆 pair            |
| `test/mock_robustness.py`         | Prompt injection 抗性 + A/B/C/D mock 场景的自检                     |
| `test/run_with_log.py`            | `run.py` 的带日志包装器（由 `run_test.sh` 调用）                    |

所有诊断脚本都不会修改考试文件，仅复用 `solution.py` 的内部方法。

### 常见问题

- **连不上 API**：先跑 `python llm_client.py`，按提示检查 Key / 网络代理。
- **Prompt 超预算警告** `[WARNING] prompt truncated by N tokens`：说明某次 `call_llm` 超出 2048，参考 [run.py:30-56](run.py#L30-L56) 的尾截断行为，正式评分系统一致，不会报错但可能影响准确率。
- **速率受限 / 超时**：降低 `--workers`；或在 [llm_client.py](llm_client.py) 里调大 `retries`。

---

## 📁 项目结构

```text
.
├── solution.py              # 核心提交文件：MyHarness 实现
├── harness_base.py          # 官方基类（不可修改）
├── llm_client.py            # LLM 调用 + tokenizer 封装
├── run.py                   # 本地评测脚本
├── run_test.sh              # 一键测试脚本
├── requirements.txt
├── data/                    # DEV / OOD / MCQ 数据集（train + test）
├── tokenizer/               # Qwen3 tokenizer 文件（本地 token 计数用）
├── test/                    # 辅助诊断脚本
│   ├── api_determinism_probe.py
│   ├── hard_case_analysis.py
│   ├── mock_robustness.py
│   └── run_with_log.py
├── output/                  # 历史运行日志与 solution 快照
├── 探索报告.pdf / .md / .tex # 探索报告（20% 主观分）
├── 任务清单.md              # 考核需求拆解
└── 探索清单.md              # 实验记录与复盘
```

---

## 🛡️ 考核硬约束

| 约束            | 值 / 规则                                              |
| --------------- | ------------------------------------------------------ |
| 模型            | Qwen3-8B Instruct（权重冻结，非思考默认关）            |
| 单 prompt token | ≤ 2048（超出尾部截断）                                 |
| 可用库          | stdlib + numpy + harness_base（禁 sklearn / torch 等） |
| 持久化          | 禁止写磁盘                                             |
| 任务类型        | 客服意图 + OOD + A/B/C/D 选择题 + 注入样本             |
| 评分            | 客观 80%（4 轮加权均值）+ 探索报告 20%                 |

---

## 📜 License

[MIT](LICENSE) © 2026 kalalsland
