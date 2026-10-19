# Best trial so far

- Study: `heretic`
- Journal: `Qwen--Qwen3--8-27B.jsonl`
- Trials completed: **12** / 13 recorded
- Baseline refusals of the original model: **90/100**

## Pareto front (sorted by refusals, then KL divergence)

| # | trial | refusals | KL divergence | direction_index |
|---:|---:|---:|---:|---|
| 0 | 5 | 67/100 | 0.0119 | `35.01889889681645` |
| 1 | 3 | 74/100 | 0.0094 | `39.566204502836634` |
| 2 | 8 | 84/100 | 0.0080 | `28.849875375041588` |
| 3 | 1 | 86/100 | 0.0015 | `48.257809167059264` |
| 4 | 10 | 88/100 | 0.0006 | `45.24692795480908` |
| 5 | 2 | 91/100 | 0.0005 | `31.888681986365704` |

## Recommended trial (front index 0)

```text
* direction_scope = per layer
* direction_index = 35.01889889681645
* attn.o_proj.max_weight = 1.1640476148244676
* attn.o_proj.max_weight_position = 51.577099039450644
* attn.o_proj.min_weight = 0.18485445552552704
* attn.o_proj.min_weight_distance = 36.680714301735755
* mlp.down_proj.max_weight = 1.3425929763527802
* mlp.down_proj.max_weight_position = 61.475373327417564
* mlp.down_proj.min_weight = 0.8948273504276488
* mlp.down_proj.min_weight_distance = 23.002719220247933
```

Refusals **67/100**, KL divergence **0.0119**.
