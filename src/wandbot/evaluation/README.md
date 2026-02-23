# Evaluation

## Overview

The following W&B reports provide an overview of the evaluation process of wandbot:


- [How to evaluate an LLM Part 1: Building an Evaluation Dataset for our LLM System](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-evaluate-an-LLM-Part-1-Building-an-Evaluation-Dataset-for-our-LLM-System--Vmlldzo1NTAwNTcy)
- [How to evaluate an LLM Part 2: Manual Evaluation of our LLM System](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-evaluate-an-LLM-Part-2-Manual-Evaluation-of-our-LLM-System--Vmlldzo1NzU4NTM3)
- [How to evaluate an LLM Part 3: Auto-Evaluation; LLMs evaluating LLMs](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-evaluate-an-LLM-Part-3-Auto-Evaluation-LLMs-evaluating-LLMs--Vmlldzo1NzEzMDcz)

## Staleness Audit Script

To audit old evaluation question/reference pairs for staleness:

```bash
python -m wandbot.evaluation.stale_eval_audit \
  --dataset weave:///wandbot/wandbot-eval/object/wandbot_eval_data:eCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU \
  --concurrency 20 \
  --model gpt-5.2 \
  --workspace-root /Users/morganmcguire/ML/wandbot \
  --codex-web-search
```

Outputs:
- `artifacts/eval_staleness/staleness_audit.jsonl` (per-row structured report)
- `artifacts/eval_staleness/staleness_summary.md` (summary counts + recommendations)

Behavior:
- Runs one headless Codex agent per question/reference row.
- Ensures local copies of `wandb/docs`, `wandb/wandb`, `wandb/weave`, and `wandb/examples` under `data/eval_staleness/repos` (unless `--no-sync-repos` is passed).
