#!/usr/bin/env bash
# Idle watcher for the AMA runs.  Every 15 minutes it appends one status line
# to logs/ama_latent_memory/monitor.log and:
#   1) marks CLEAN_DONE once the clean arm's evaluation.json exists;
#   2) launches the latent+anchor GPU smoke once (4 questions, element style)
#      when any GPU 0-5 has <3 GB used and the clean answer process has exited.
# Run with: nohup bash xt_ama_adapter/scripts/monitor_ama_runs.sh >/dev/null 2>&1 &
set -u

RUN_ROOT=/home/cbd/project/residual-mem
LOG="$RUN_ROOT/logs/ama_latent_memory/monitor.log"
PY32="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
CLEAN_K5="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-clean/k5"
ANCHOR_K5="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-anchor/element/k5"
ANCHOR_K8="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-anchor/element/k8"
ANCHOR_H8="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-anchor/element/k8-hybrid"
AUG10="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-anchor/element/k10-augment"
JUDGE_SH="$RUN_ROOT/xt_ama_adapter/scripts/run_ama_anchor_element_judge.sh"
SMOKE_LOG="$RUN_ROOT/logs/ama_latent_memory/anchor-smoke.log"
SMOKE_CKPT="$RUN_ROOT/logs/ama_latent_memory/anchor-smoke-checkpoint.jsonl"
SMOKE_ANSWERS="$RUN_ROOT/logs/ama_latent_memory/anchor-smoke-answers.jsonl"
MARK="$RUN_ROOT/logs/ama_latent_memory/anchor_smoke_launched"
mkdir -p "$(dirname "$LOG")"
touch "$LOG"
# Single-instance guard: a second watcher simply exits.
exec 9>"$LOG.lock"
flock -n 9 || exit 0
echo $$ > "$LOG.pid"

smoke_launched=0
clean_reported=0
h8judge_launched=0
aug10_launched=0
[ -f "$MARK" ] && smoke_launched=1
while true; do
  ts=$(date --iso-8601=seconds)
  ckpt=$(wc -l < "$CLEAN_K5/answer-checkpoint.jsonl" 2>/dev/null || echo 0)
  answers=$(wc -l < "$CLEAN_K5/answers.jsonl" 2>/dev/null || echo 0)
  ev="no"; [ -f "$CLEAN_K5/evaluation.json" ] && ev="yes"
  acc="-"
  if [[ "$ev" == "yes" ]]; then
    acc=$("$PY32" -c "import json;print(json.load(open('$CLEAN_K5/evaluation.json'))['overall']['accuracy'])" 2>/dev/null || echo "err")
  fi
  ackpt=$(wc -l < "$ANCHOR_K5/answer-checkpoint.jsonl" 2>/dev/null || echo 0)
  aanswers=$(wc -l < "$ANCHOR_K5/answers.jsonl" 2>/dev/null || echo 0)
  aev="no"; [ -f "$ANCHOR_K5/evaluation.json" ] && aev="yes"
  aacc="-"
  if [[ "$aev" == "yes" ]]; then
    aacc=$("$PY32" -c "import json;print(json.load(open('$ANCHOR_K5/evaluation.json'))['overall']['accuracy'])" 2>/dev/null || echo "err")
  fi
  k8ckpt=$(wc -l < "$ANCHOR_K8/answer-checkpoint.jsonl" 2>/dev/null || echo 0)
  k8answers=$(wc -l < "$ANCHOR_K8/answers.jsonl" 2>/dev/null || echo 0)
  k8ev="no"; [ -f "$ANCHOR_K8/evaluation.json" ] && k8ev="yes"
  k8acc="-"
  if [[ "$k8ev" == "yes" ]]; then
    k8acc=$("$PY32" -c "import json;print(json.load(open('$ANCHOR_K8/evaluation.json'))['overall']['accuracy'])" 2>/dev/null || echo "err")
  fi
  h8ckpt=$(wc -l < "$ANCHOR_H8/answer-checkpoint.jsonl" 2>/dev/null || echo 0)
  h8answers=$(wc -l < "$ANCHOR_H8/answers.jsonl" 2>/dev/null || echo 0)
  h8ev="no"; [ -f "$ANCHOR_H8/evaluation.json" ] && h8ev="yes"
  h8acc="-"
  if [[ "$h8ev" == "yes" ]]; then
    h8acc=$("$PY32" -c "import json;print(json.load(open('$ANCHOR_H8/evaluation.json'))['overall']['accuracy'])" 2>/dev/null || echo "err")
  fi
  gpu="(nvidia-smi disabled under sandbox)"
  echo "$ts clean_ckpt=$ckpt clean_answers=$answers eval=$ev acc=$acc anchor_ckpt=$ackpt anchor_answers=$aanswers anchor_eval=$aev anchor_acc=$aacc anchor8_ckpt=$k8ckpt anchor8_answers=$k8answers anchor8_eval=$k8ev anchor8_acc=$k8acc hybrid8_ckpt=$h8ckpt hybrid8_answers=$h8answers hybrid8_eval=$h8ev hybrid8_acc=$h8acc gpu=[$gpu]" >> "$LOG"

  if [[ "$ev" == "yes" && "$clean_reported" -eq 0 ]]; then
    clean_reported=1
    echo "$ts CLEAN_DONE accuracy=$acc (evaluation.json present)" >> "$LOG"
  fi

  # Auto-judge for the hybrid top-8 arm once its answers are on disk and the
  # answer process has exited.  Idempotent: evaluation.json present, an active
  # judge script, or our own launch flag all suppress a second launch.
  if [[ "$h8ev" == "no" && "$h8judge_launched" -eq 0 ]] \
     && [[ -s "$ANCHOR_H8/answers.jsonl" ]] \
     && ! pgrep -f "run_ama_web_latent answer" >/dev/null \
     && ! pgrep -f "run_ama_anchor_element_judge.sh" >/dev/null; then
    h8judge_launched=1
    echo "$ts HYBRID8_JUDGE_LAUNCHED (JDIR=element/k8-hybrid)" >> "$LOG"
    JDIR=element/k8-hybrid nohup bash "$JUDGE_SH" \
      > "$RUN_ROOT/logs/ama_latent_memory/ama_eval/latent-only-top10-anchor/judge-k8-hybrid-runner.log" 2>&1 &
  fi

  # Auto-judge for the k10-augment (option B) arm, same idempotence rules.
  if [[ "$aug10_launched" -eq 0 && ! -f "$AUG10/evaluation.json" ]] \
     && [[ -s "$AUG10/answers.jsonl" ]] \
     && ! pgrep -f "run_ama_web_latent answer" >/dev/null \
     && ! pgrep -f "run_ama_anchor_element_judge.sh" >/dev/null; then
    aug10_launched=1
    echo "$ts AUG10_JUDGE_LAUNCHED (JDIR=element/k10-augment)" >> "$LOG"
    JDIR=element/k10-augment nohup bash "$JUDGE_SH" \
      > "$RUN_ROOT/logs/ama_latent_memory/ama_eval/latent-only-top10-anchor/judge-k10-augment-runner.log" 2>&1 &
  fi

  # GPU auto-smoke is only re-enabled explicitly (AUTO_SMOKE=1); the anchor GPU
  # smoke already ran once and its marker is set.  Skipping nvidia-smi keeps
  # this watcher fully filesystem-only so it survives the tool sandbox.
  if [[ "${AUTO_SMOKE:-0}" == "1" && "$smoke_launched" -eq 0 && ! -f "$MARK" ]]; then
    free_gpu=""
    while IFS=', ' read -r idx used util; do
      used=${used% MiB}; util=${util% %}
      if [[ "$idx" -lt 6 && "${used:-999999}" -lt 3000 ]]; then
        free_gpu=$idx; break
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)
    if [[ -n "$free_gpu" ]] && ! pgrep -f "run_ama_web_latent answer" >/dev/null; then
      touch "$MARK"
      echo "$ts SMOKE_LAUNCHED gpu=$free_gpu (latent+anchor element, limit 4)" >> "$LOG"
      CUDA_VISIBLE_DEVICES="$free_gpu" PYTHONPATH=.:xt_ama_adapter \
        "$PY32" -u -m experiments.state_tokenizer.run_ama_web_latent answer \
          --test-file "$RUN_ROOT/third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl" \
          --cache-dir "$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/cache-v2" \
          --qformer "$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt" \
          --retrieval-head "$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt" \
          --bridge "$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-top10.pt" \
          --reader-model /data1/models/Qwen3-32B \
          --checkpoint "$SMOKE_CKPT" --answers-output "$SMOKE_ANSWERS" \
          --memory-mode latent+anchor --anchor-style element \
          --top-k-latent 5 --top-k 5 \
          --enable-thinking true --max-model-len 32000 --max-new-tokens 8192 \
          --batch-size 1 --max-memory-gib 70 --skip-episode-ids 184 --limit 4 \
          > "$SMOKE_LOG" 2>&1 &
      smoke_launched=1
      echo "$ts smoke_launched=$smoke_launched" >> "$LOG"
    fi
  fi
  sleep 900
done
