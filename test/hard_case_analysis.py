"""
hard_case_analysis.py
=====================
困难样本分析：跑完整 dev，每条样本记录
- 检索 top-24 的 label 序列
- gold 是否在 top-1 / top-5 / top-10 / top-24 内（检索 recall）
- LLM 预测结果与是否正确

随后输出统计：
- 检索 recall 分布
- 错误样本里 gold 在检索范围的比例 → 区分"检索 miss"vs"LLM 决策错"
- 混淆 pair（gold ↔ pred）
- 错误聚集的 gold label / 模型最爱预测错的 label

不修改 solution.py / run.py / harness_base.py。复用 harness 内部 _search 等方法。
输出到 ../../output/hardcase_<时间戳>/。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAM_DIR = HERE.parent
OUTPUT_ROOT = EXAM_DIR.parent / "output"

sys.path.insert(0, str(EXAM_DIR))

from solution import MyHarness  # noqa: E402
from llm_client import (  # noqa: E402
    call_llm as _raw_call_llm,
    count_tokens,
    count_messages_tokens,
    truncate_to_tokens,
)


def make_controlled_llm(max_prompt_tokens, tracker, lock):
    """复刻 run.py:30-56 的 prompt 截断 + token 统计。"""
    def _call(messages):
        prompt_text = " ".join(m.get("content", "") for m in messages)
        n = count_tokens(prompt_text)
        if n > max_prompt_tokens:
            messages = list(messages)
            excess = n - max_prompt_tokens
            for i in range(len(messages) - 1, -1, -1):
                if excess <= 0:
                    break
                content = messages[i].get("content", "")
                msg_tokens = count_tokens(content)
                if msg_tokens <= excess:
                    messages[i] = {**messages[i], "content": ""}
                    excess -= msg_tokens
                else:
                    messages[i] = {**messages[i], "content": truncate_to_tokens(content, msg_tokens - excess)}
                    excess = 0
            n = count_tokens(" ".join(m.get("content", "") for m in messages))
        resp = _raw_call_llm(messages)
        with lock:
            tracker["prompt"] += n
            tracker["completion"] += count_tokens(resp)
        return resp
    return _call


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    train = load_jsonl(EXAM_DIR / "data" / "train_dev.jsonl")
    dev = load_jsonl(EXAM_DIR / "data" / "test_dev.jsonl")

    tracker = {"prompt": 0, "completion": 0}
    lock = threading.Lock()
    llm = make_controlled_llm(2048, tracker, lock)

    harness = MyHarness(llm, count_tokens, count_messages_tokens, 2048)
    for item in train:
        harness.update(item["text"], item["label"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"hardcase_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hardcase] Train={len(train)}  Dev={len(dev)}")
    print(f"[hardcase] Output: {out_dir}\n")

    # 提前算所有 dev 的检索 top-K（不烧 API）
    pre_records = []
    for i, item in enumerate(dev):
        text = item["text"]
        gold = item["label"]
        topk_idx = harness._search(text, k=24)
        topk_labels = [harness.memory[d][1] for d in topk_idx]
        pre_records.append({
            "idx": i,
            "text": text,
            "gold": gold,
            "topk_labels": topk_labels,
        })

    # 并行调 predict 收集预测
    workers = 8
    print(f"[hardcase] running predict with workers={workers}...")
    t0 = time.time()
    preds = [None] * len(dev)
    errors_call = []

    def run_one(i):
        try:
            return i, harness.predict(dev[i]["text"]), None
        except Exception as e:
            return i, "", str(e)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, i) for i in range(len(dev))]
        done = 0
        for fut in as_completed(futs):
            i, pred, err = fut.result()
            preds[i] = pred
            if err:
                errors_call.append((i, err))
            done += 1
            sys.stdout.write(f"\r[hardcase]   progress: {done}/{len(dev)}")
            sys.stdout.flush()

    print(f"\n[hardcase] elapsed: {time.time() - t0:.1f}s")
    if errors_call:
        print(f"[hardcase] LLM call errors: {len(errors_call)}")

    # 合并诊断信息
    records = []
    for r, pred in zip(pre_records, preds):
        labs = r["topk_labels"]
        gold = r["gold"]
        records.append({
            **r,
            "pred": pred,
            "correct": pred == gold,
            "gold_in_top1": gold == (labs[0] if labs else None),
            "gold_in_top5": gold in set(labs[:5]),
            "gold_in_top10": gold in set(labs[:10]),
            "gold_in_top24": gold in set(labs),
            "n_retrieved": len(labs),
        })

    # 落详细记录
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    total = len(records)
    correct = sum(r["correct"] for r in records)
    errors = [r for r in records if not r["correct"]]

    n_top1 = sum(r["gold_in_top1"] for r in records)
    n_top5 = sum(r["gold_in_top5"] for r in records)
    n_top10 = sum(r["gold_in_top10"] for r in records)
    n_top24 = sum(r["gold_in_top24"] for r in records)

    err_top1 = sum(r["gold_in_top1"] for r in errors)
    err_top5 = sum(r["gold_in_top5"] for r in errors)
    err_top10 = sum(r["gold_in_top10"] for r in errors)
    err_top24 = sum(r["gold_in_top24"] for r in errors)
    err_no_retrieve = sum(1 for r in errors if r["n_retrieved"] == 0)

    confusion = Counter((r["gold"], r["pred"]) for r in errors)
    err_by_gold = Counter(r["gold"] for r in errors)
    err_by_pred = Counter(r["pred"] for r in errors)

    # 按 gold 标签拆错误率（要求 gold 出现 ≥ 2 次才算）
    gold_total = Counter(r["gold"] for r in records)
    label_err_rate = []
    for lab, n_err in err_by_gold.most_common():
        n_tot = gold_total.get(lab, 0)
        if n_tot >= 2:
            label_err_rate.append((lab, n_err, n_tot, n_err / n_tot))

    # 写 markdown 摘要
    md = []
    md.append(f"# Hard case analysis — {datetime.now().isoformat(timespec='seconds')}")
    md.append("")
    md.append(f"- Total dev: **{total}**")
    md.append(f"- Correct: **{correct} ({correct/total*100:.1f}%)**")
    md.append(f"- Errors: **{len(errors)} ({len(errors)/total*100:.1f}%)**")
    md.append(f"- Token usage: prompt={tracker['prompt']:,}, completion={tracker['completion']:,}")
    md.append("")
    md.append("## 一、检索 recall —— gold 是否被检索到？")
    md.append("")
    md.append("整 dev 集（含正确样本）的检索命中：")
    md.append("")
    md.append("| 范围 | 命中数 | 覆盖率 |")
    md.append("|---|---|---|")
    md.append(f"| gold == retrieval top-1 | {n_top1} | {n_top1/total*100:.1f}% |")
    md.append(f"| gold ∈ retrieval top-5 | {n_top5} | {n_top5/total*100:.1f}% |")
    md.append(f"| gold ∈ retrieval top-10 | {n_top10} | {n_top10/total*100:.1f}% |")
    md.append(f"| gold ∈ retrieval top-24 | {n_top24} | {n_top24/total*100:.1f}% |")
    md.append("")
    md.append("## 二、错误样本中检索分布 —— 区分 \"检索 miss\" vs \"LLM 决策错\"")
    md.append("")
    md.append(f"错误总数: **{len(errors)}**")
    md.append("")
    md.append("| 状态 | 计数 | 占错误的比例 |")
    md.append("|---|---|---|")
    md.append(f"| gold == top-1（检索一击命中但 LLM 选错） | {err_top1} | {err_top1/len(errors)*100:.1f}% |")
    md.append(f"| gold ∈ top-5 | {err_top5} | {err_top5/len(errors)*100:.1f}% |")
    md.append(f"| gold ∈ top-10 | {err_top10} | {err_top10/len(errors)*100:.1f}% |")
    md.append(f"| gold ∈ top-24（gold 在检索池内 → LLM 决策问题） | {err_top24} | {err_top24/len(errors)*100:.1f}% |")
    md.append(f"| gold ∉ top-24（检索完全 miss） | {len(errors) - err_top24} | {(len(errors)-err_top24)/len(errors)*100:.1f}% |")
    md.append(f"| 检索 0 命中（无候选可选） | {err_no_retrieve} | {err_no_retrieve/len(errors)*100:.1f}% |")
    md.append("")
    md.append("## 三、Top 30 混淆对（gold → pred）")
    md.append("")
    md.append("| gold | pred | count |")
    md.append("|---|---|---|")
    for (g, p), c in confusion.most_common(30):
        md.append(f"| `{g}` | `{p}` | {c} |")
    md.append("")
    md.append("## 四、错误聚集的 gold label（按错误率排序，要求 gold 在 dev 出现 ≥ 2 次）")
    md.append("")
    md.append("| gold label | err / total | err rate |")
    md.append("|---|---|---|")
    label_err_rate.sort(key=lambda x: (-x[3], -x[1]))
    for lab, e, t, rate in label_err_rate[:30]:
        md.append(f"| `{lab}` | {e}/{t} | {rate*100:.0f}% |")
    md.append("")
    md.append("## 五、模型最爱预测的错误 label（top 15）")
    md.append("")
    md.append("| label | wrong-pred count |")
    md.append("|---|---|")
    for lab, c in err_by_pred.most_common(15):
        md.append(f"| `{lab}` | {c} |")
    md.append("")

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    # 单独导出错误样本（带检索 trace）
    err_export = []
    for r in errors:
        err_export.append({
            "idx": r["idx"],
            "text": r["text"],
            "gold": r["gold"],
            "pred": r["pred"],
            "gold_in_top1": r["gold_in_top1"],
            "gold_in_top5": r["gold_in_top5"],
            "gold_in_top10": r["gold_in_top10"],
            "gold_in_top24": r["gold_in_top24"],
            "topk_labels": r["topk_labels"][:10],
        })
    with open(out_dir / "errors.jsonl", "w", encoding="utf-8") as f:
        for r in err_export:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[hardcase] DEV accuracy: {correct/total*100:.1f}%")
    print(f"[hardcase] errors with gold in top-1: {err_top1}")
    print(f"[hardcase] errors with gold in top-5: {err_top5}")
    print(f"[hardcase] errors with gold ∉ top-24: {len(errors)-err_top24}")
    print(f"[hardcase] summary: {out_dir/'summary.md'}")
    print(f"[hardcase] errors.jsonl: {out_dir/'errors.jsonl'}")


if __name__ == "__main__":
    main()
