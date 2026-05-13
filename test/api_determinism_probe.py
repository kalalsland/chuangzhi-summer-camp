"""
api_determinism_probe.py
=========================
30 秒诊断当前配置的 LLM API 是否 deterministic（同 prompt 多次输出是否完全一致）。

用法
----
1. 切换 llm_client.py 里的 BASE_URL / API_KEY / MODEL 到目标服务（DashScope / 硅基流动 /
   本地 vLLM ...）
2. python test/api_determinism_probe.py

输出
----
- 5 次同 prompt 的 completion 是否字面完全一致
- 每次的 token 数是否相同
- 给出 "投票有效性" 判断

判断逻辑
----
- 5 次输出完全一致 → deterministic，自一致性投票无效（DashScope 已观察到）
- 5 次输出有差异   → 真采样，投票可压方差，N=3 大概率正向
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAM_DIR = HERE.parent
sys.path.insert(0, str(EXAM_DIR))

from llm_client import call_llm, count_tokens, OPENAI_CONFIG  # noqa: E402


PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a strict text classifier. Pick exactly one label from "
            "LABEL_SET to classify the text. Output only the label."
        ),
    },
    {
        "role": "user",
        "content": (
            "LABEL_SET: card_arrival, card_delivery_estimate, balance, transfer_timing, "
            "exchange_rate, declined_card_payment, request_refund, top_up_failed\n\n"
            "Text: \"My card payment did not work, what happened?\"\n\n"
            "Output the chosen label only."
        ),
    },
]

N_TRIALS = 5


def main():
    print(f"[probe] base_url:  {OPENAI_CONFIG['base_url']}")
    print(f"[probe] model:     {OPENAI_CONFIG['model']}")
    print(f"[probe] temp/top_p:{OPENAI_CONFIG['temperature']} / {OPENAI_CONFIG['top_p']}")
    print(f"[probe] running {N_TRIALS} identical calls...\n")

    results = []
    for i in range(N_TRIALS):
        try:
            resp = call_llm(PROMPT)
        except Exception as e:
            print(f"  call #{i+1}: ERROR — {e}")
            return 1
        n_tok = count_tokens(resp)
        results.append((resp, n_tok))
        print(f"  call #{i+1}: tokens={n_tok}, text={resp.strip()[:80]!r}")

    print()
    unique_texts = {r for r, _ in results}
    unique_tokens = {n for _, n in results}

    print("=" * 70)
    if len(unique_texts) == 1 and len(unique_tokens) == 1:
        print("[probe] VERDICT: DETERMINISTIC (5/5 identical outputs)")
        print("[probe] → 同 prompt 多次输出完全一致 → 服务端疑似贪心解码 / prompt cache")
        print("[probe] → 自一致性投票 (_N_VOTES > 1) 在此 API 上无效")
        print("[probe] → 但评测方部署可能不同；任务清单提到'4 次采样取均值',")
        print("[probe]   暗示评测方真采样, 投票仍可能在评测时有效")
    elif len(unique_texts) == 1:
        print("[probe] VERDICT: TEXT-DETERMINISTIC")
        print(f"[probe] → 5 次输出文本一致, 但 token 数不同 ({sorted(unique_tokens)})")
        print("[probe] → 边界 case, 投票仍倾向无效")
    else:
        print(f"[probe] VERDICT: STOCHASTIC ({len(unique_texts)} distinct outputs in {N_TRIALS})")
        print("[probe] → 同 prompt 不同输出 → 真采样有效")
        print("[probe] → 自一致性投票 (_N_VOTES = 3) 大概率正向收益, 推荐启用")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
