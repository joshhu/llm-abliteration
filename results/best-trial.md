# Best trial so far

- Study: `heretic`
- Journal: `Qwen--Qwen3--8-27B.jsonl`
- Trials completed: **85** / 86 recorded
- Baseline refusals of the original model: **90/100**

## Pareto front (sorted by refusals, then KL divergence)

| # | trial | refusals | KL divergence | direction_index |
|---:|---:|---:|---:|---|
| 0 | 33 | 61/100 | 0.0098 | `55.969831945361435` |
| 1 | 34 | 65/100 | 0.0085 | `53.39622037074517` |
| 2 | 82 | 69/100 | 0.0075 | `52.99126519254025` |
| 3 | 13 | 70/100 | 0.0069 | `54.901655673244356` |
| 4 | 31 | 72/100 | 0.0063 | `54.13560650188851` |
| 5 | 43 | 74/100 | 0.0057 | `56.16082935376518` |
| 6 | 25 | 76/100 | 0.0056 | `47.15599584135094` |
| 7 | 85 | 79/100 | 0.0031 | `54.63083644401726` |
| 8 | 15 | 82/100 | 0.0012 | `46.37226974328517` |
| 9 | 24 | 84/100 | 0.0002 | `42.739239767438036` |

## Recommended trial (front index 0)

```text
* direction_scope = per layer
* direction_index = 55.969831945361435
* attn.o_proj.max_weight = 1.1088403439079244
* attn.o_proj.max_weight_position = 52.078190895907234
* attn.o_proj.min_weight = 0.6331014359410128
* attn.o_proj.min_weight_distance = 21.308076951480082
* mlp.down_proj.max_weight = 1.4051763834634836
* mlp.down_proj.max_weight_position = 60.830261688695124
* mlp.down_proj.min_weight = 0.23346587991950035
* mlp.down_proj.min_weight_distance = 23.338562400519766
```

Refusals **61/100**, KL divergence **0.0098**.
