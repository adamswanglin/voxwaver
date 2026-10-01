#!/bin/zsh
# Sample RSS of $1 every 0.3s until exit; log timeline.
pid=$1
log=/tmp/mem_timeline.log
: > $log
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
  echo "$(date +%s.%N) rss_mb=$(( ${rss:-0} / 1024 ))" >> $log
  sleep 0.3
done
echo "--- top 6 RSS samples (MB) ---"
awk '{print $2}' $log | sort -t= -k2 -rn | head -6
