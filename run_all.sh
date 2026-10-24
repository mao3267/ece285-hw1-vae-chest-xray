#!/usr/bin/env bash
# Train and evaluate the beta sweep. Usage: ./run_all.sh [train|eval|all]   (default: all)
# Finished runs are skipped, so it is safe to re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")"

STAGE="${1:-all}"
BETAS=(1 0.1 0.01)

for beta in "${BETAS[@]}"; do
  name="beta${beta}"
  if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
    if [[ -f "runs/${name}/last.pt" && "$(tail -n1 "runs/${name}/metrics.csv" | cut -d, -f1)" == "100" ]]; then
      echo "skip training ${name} (finished)"
    else
      uv run python -m vae_xray.train --name "${name}" --beta "${beta}" --force
    fi
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    if [[ -f "runs/${name}/eval.json" ]]; then
      echo "skip evaluation ${name} (eval.json exists)"
    else
      uv run python -m vae_xray.evaluate --run "${name}"
    fi
  fi
done
