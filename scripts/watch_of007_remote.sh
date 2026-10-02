set -e
end=$(date -d '2026-09-26T06:48:06Z' +%s)
while [ "$(date +%s)" -lt "$end" ]; do
  tail -n 2 /workspace/outputs/OF007-run.log
  if test -f /workspace/outputs/OF007/SUITE_COMPLETED && test -f /workspace/outputs/OF007-pip_freeze.txt; then
    python3 /workspace/motion-valley/scripts/collect_of007_evidence.py
    exit 0
  fi
  if ! kill -0 "$(cat /workspace/outputs/OF007-launch-lock/pid)" 2>/dev/null; then
    tail -n 20 /workspace/outputs/OF007-run.log
    exit 2
  fi
  sleep 30
done
exit 3
