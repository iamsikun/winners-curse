#!/usr/bin/env bash
# Run every A/B simulation YAML, sequentially, from any working directory.
set -euo pipefail

dry_run=false
case "${1:-}" in
  --dry-run) dry_run=true; shift ;;
  --help|-h)
    printf 'Usage: %s [--dry-run]\nRuns every configs/ab_test_*.yaml simulation.\n' "$0"
    exit 0 ;;
  '') ;;
  *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
esac
if (( $# != 0 )); then
  printf 'Unexpected arguments: %s\n' "$*" >&2
  exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

configs=(configs/ab_test_*.yaml)

if [[ "$dry_run" == true ]]; then
  for config in "${configs[@]}"; do
    printf '%q ' uv run python -u scripts/ab_test_experiment.py --config "$config"
    printf '\n'
  done
  exit 0
fi

batch_started=$SECONDS
completed=0
current_config=''
finished_configs=()
finished_seconds=()

print_summary() {
  local status=$?
  local index
  printf '\nBatch summary: %d/%d completed; elapsed %ds\n' \
    "$completed" "${#configs[@]}" "$((SECONDS - batch_started))"
  for ((index=0; index<completed; index++)); do
    printf '  DONE %s (%ss)\n' "${finished_configs[index]}" "${finished_seconds[index]}"
  done
  if [[ -n "$current_config" ]]; then
    if (( status == 130 || status == 143 )); then
      printf '  INTERRUPTED %s (%ds)\n' "$current_config" "$((SECONDS - run_started))"
    else
      printf '  FAILED %s (exit %d, %ds)\n' "$current_config" "$status" "$((SECONDS - run_started))"
    fi
    printf '  %d experiment(s) not started.\n' "$((${#configs[@]} - completed - 1))"
  fi
}
trap print_summary EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Starting batch: %d experiments. Progress is counted by completed configs.\n' "${#configs[@]}"
for config in "${configs[@]}"; do
  current_config=$config
  run_started=$SECONDS
  printf '\n[batch %d/%d | %d%% complete] START %s\n' \
    "$((completed + 1))" "${#configs[@]}" "$((100 * completed / ${#configs[@]}))" "$config"
  command=(uv run python -u scripts/ab_test_experiment.py --config "$config")
  printf '%q ' "${command[@]}"
  printf '\n'
  if "${command[@]}"; then
    finished_configs+=("$config")
    finished_seconds+=("$((SECONDS - run_started))")
    completed=$((completed + 1))
    printf '[batch %d/%d | %d%% complete] DONE %s | experiment %ds | batch elapsed %ds\n' \
      "$completed" "${#configs[@]}" "$((100 * completed / ${#configs[@]}))" \
      "$config" "$((SECONDS - run_started))" "$((SECONDS - batch_started))"
    current_config=''
  else
    status=$?
    exit "$status"
  fi
done
