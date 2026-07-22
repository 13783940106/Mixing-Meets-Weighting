# Patch Plan

1. Add one formal switch `use_emr_adamix` and mutual-exclusion checks.
2. Reuse EMA source logits to compute detached entropy-margin reliability.
3. Reuse the exact historical AdaMix mask generator.
4. Extend `mix_loss` only through optional reliability masks; default-off executes the historical branch.
5. Route weak/strong A/B reliability to the four matching mixed losses.
6. Add mechanism endpoint logging; do not modify evaluator/checkpoint/optimizer/scheduler/EMA.
