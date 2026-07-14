#!/bin/sh
# Bring up the whole stack inside one container, because the free-tier hosts run
# exactly one. Compose still owns local dev; this file exists only for the demo.
set -e

redis-server --daemonize yes --save '' --appendonly no

# No persistent disk anywhere this runs, so the collection is rebuilt from kb/ on
# every start. That is a few seconds of embedding on a corpus this size, and it
# means there is no stale-index failure mode to debug at 3am.
#
# On Cloud Run the filesystem is a tmpfs, so this index lives in RAM and counts
# against the memory limit. 25 documents is a few MB; if kb/ ever grows by an order
# of magnitude, that is the number to watch.
QDRANT__STORAGE__STORAGE_PATH=/tmp/qdrant/storage \
QDRANT__STORAGE__SNAPSHOTS_PATH=/tmp/qdrant/snapshots \
QDRANT__TELEMETRY_DISABLED=true \
  qdrant &

until curl -fsS http://localhost:6333/readyz >/dev/null 2>&1; do sleep 0.5; done

python scripts/ingest.py

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
