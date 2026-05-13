# Hard case analysis — 2026-05-07T21:15:23

- Total dev: **539**
- Correct: **437 (81.1%)**
- Errors: **102 (18.9%)**
- Token usage: prompt=384,798, completion=2,052

## 一、检索 recall —— gold 是否被检索到？

整 dev 集（含正确样本）的检索命中：

| 范围 | 命中数 | 覆盖率 |
|---|---|---|
| gold == retrieval top-1 | 255 | 47.3% |
| gold ∈ retrieval top-5 | 414 | 76.8% |
| gold ∈ retrieval top-10 | 465 | 86.3% |
| gold ∈ retrieval top-24 | 501 | 92.9% |

## 二、错误样本中检索分布 —— 区分 "检索 miss" vs "LLM 决策错"

错误总数: **102**

| 状态 | 计数 | 占错误的比例 |
|---|---|---|
| gold == top-1（检索一击命中但 LLM 选错） | 14 | 13.7% |
| gold ∈ top-5 | 32 | 31.4% |
| gold ∈ top-10 | 51 | 50.0% |
| gold ∈ top-24（gold 在检索池内 → LLM 决策问题） | 75 | 73.5% |
| gold ∉ top-24（检索完全 miss） | 27 | 26.5% |
| 检索 0 命中（无候选可选） | 0 | 0.0% |

## 三、Top 30 混淆对（gold → pred）

| gold | pred | count |
|---|---|---|
| `card_arrival` | `card_delivery_estimate` | 4 |
| `get_physical_card` | `change_pin` | 3 |
| `order_physical_card` | `card_delivery_estimate` | 3 |
| `wrong_exchange_rate_for_cash_withdrawal` | `exchange_rate` | 3 |
| `balance_not_updated_after_bank_transfer` | `transfer_timing` | 3 |
| `topping_up_by_card` | `transfer_into_account` | 2 |
| `request_refund` | `cancel_transfer` | 2 |
| `top_up_failed` | `declined_card_payment` | 2 |
| `top_up_by_card_charge` | `top_up_limits` | 2 |
| `getting_spare_card` | `card_linking` | 2 |
| `virtual_card_not_working` | `declined_card_payment` | 2 |
| `top_up_by_bank_transfer_charge` | `transfer_fee_charged` | 2 |
| `why_verify_identity` | `unable_to_verify_identity` | 2 |
| `unable_to_verify_identity` | `verify_my_identity` | 2 |
| `transfer_not_received_by_recipient` | `pending_transfer` | 2 |
| `supported_cards_and_currencies` | `card_acceptance` | 2 |
| `beneficiary_not_allowed` | `declined_transfer` | 2 |
| `reverted_card_payment?` | `declined_card_payment` | 2 |
| `direct_debit_payment_not_recognised` | `pending_cash_withdrawal` | 1 |
| `declined_card_payment` | `card_payment_not_recognised` | 1 |
| `transfer_into_account` | `receiving_money` | 1 |
| `extra_charge_on_statement` | `reverted_card_payment?` | 1 |
| `extra_charge_on_statement` | `transfer_timing` | 1 |
| `visa_or_mastercard` | `supported_cards_and_currencies` | 1 |
| `top_up_by_card_charge` | `exchange_charge` | 1 |
| `direct_debit_payment_not_recognised` | `transaction_charged_twice` | 1 |
| `wrong_amount_of_cash_received` | `declined_cash_withdrawal` | 1 |
| `topping_up_by_card` | `pending_top_up` | 1 |
| `supported_cards_and_currencies` | `top_up_by_card_charge` | 1 |
| `receiving_money` | `fiat_currency_support` | 1 |

## 四、错误聚集的 gold label（按错误率排序，要求 gold 在 dev 出现 ≥ 2 次）

| gold label | err / total | err rate |
|---|---|---|
| `wrong_exchange_rate_for_cash_withdrawal` | 6/7 | 86% |
| `topping_up_by_card` | 5/7 | 71% |
| `supported_cards_and_currencies` | 5/7 | 71% |
| `get_physical_card` | 4/7 | 57% |
| `top_up_by_card_charge` | 4/7 | 57% |
| `card_arrival` | 4/7 | 57% |
| `balance_not_updated_after_bank_transfer` | 4/7 | 57% |
| `beneficiary_not_allowed` | 4/7 | 57% |
| `direct_debit_payment_not_recognised` | 3/7 | 43% |
| `order_physical_card` | 3/7 | 43% |
| `why_verify_identity` | 3/7 | 43% |
| `getting_spare_card` | 3/7 | 43% |
| `get_disposable_virtual_card` | 3/7 | 43% |
| `declined_transfer` | 3/7 | 43% |
| `cash_withdrawal_not_recognised` | 3/7 | 43% |
| `transfer_not_received_by_recipient` | 3/7 | 43% |
| `extra_charge_on_statement` | 2/7 | 29% |
| `request_refund` | 2/7 | 29% |
| `receiving_money` | 2/7 | 29% |
| `top_up_failed` | 2/7 | 29% |
| `exchange_via_app` | 2/7 | 29% |
| `pending_cash_withdrawal` | 2/7 | 29% |
| `virtual_card_not_working` | 2/7 | 29% |
| `top_up_by_bank_transfer_charge` | 2/7 | 29% |
| `card_payment_fee_charged` | 2/7 | 29% |
| `unable_to_verify_identity` | 2/7 | 29% |
| `reverted_card_payment?` | 2/7 | 29% |
| `declined_card_payment` | 1/7 | 14% |
| `transfer_into_account` | 1/7 | 14% |
| `visa_or_mastercard` | 1/7 | 14% |

## 五、模型最爱预测的错误 label（top 15）

| label | wrong-pred count |
|---|---|
| `card_delivery_estimate` | 8 |
| `declined_card_payment` | 7 |
| `transfer_timing` | 4 |
| `exchange_rate` | 4 |
| `pending_transfer` | 4 |
| `card_payment_not_recognised` | 3 |
| `change_pin` | 3 |
| `transfer_into_account` | 3 |
| `fiat_currency_support` | 3 |
| `card_linking` | 3 |
| `verify_my_identity` | 3 |
| `declined_transfer` | 3 |
| `card_acceptance` | 3 |
| `receiving_money` | 2 |
| `supported_cards_and_currencies` | 2 |
