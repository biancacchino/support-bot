#!/bin/sh
# Deploy the demo to Cloud Run.
#
#   GEMINI_API_KEY=... ADMIN_API_KEY=... scripts/deploy_cloudrun.sh
#
# Prerequisites, once: a Google Cloud project with billing enabled, `gcloud auth
# login`, and `gcloud config set project <id>`. Cloud Run's free tier covers a demo
# with room to spare, but the project still needs a card on file to exist.
#
# Cloud Build builds from a directory, and it builds the Dockerfile at the root of
# it. Ours is not at the root - the root Dockerfile is the compose one, which expects
# Qdrant and Redis to be other containers, and deploying that would produce a service
# that starts and then fails every request. So this stages the tree in a temp dir with
# the single-container Dockerfile in the right place, and builds that.
set -eu

: "${GEMINI_API_KEY:?set GEMINI_API_KEY}"
: "${ADMIN_API_KEY:?set ADMIN_API_KEY - /admin/metrics is open without it}"

SERVICE=${SERVICE:-support-bot}
REGION=${REGION:-us-central1}

root=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT

# Tracked files only. .env cannot be shipped even by accident.
git -C "$root" archive HEAD | tar -x -C "$staging"
cp "$root/deploy/single/Dockerfile" "$staging/Dockerfile"

cd "$staging"

# Build and deploy are two commands, not `--source`, for one reason: `--source` builds
# on Cloud Build's 10-minute default timeout, and this image cannot be built in ten
# minutes. Installing torch and baking two models in takes longer than that, and the
# failure is a bare DEADLINE_EXCEEDED that says nothing about which of the two clocks
# ran out. Naming the timeout makes the constraint visible instead of mysterious.
PROJECT=$(gcloud config get-value project 2>/dev/null)
IMAGE="$REGION-docker.pkg.dev/$PROJECT/cloud-run-source-deploy/$SERVICE"

gcloud builds submit . --region "$REGION" --timeout 30m --tag "$IMAGE"

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 120 \
  --startup-probe "httpGet.path=/health,initialDelaySeconds=20,periodSeconds=10,failureThreshold=12" \
  --set-env-vars "GEMINI_API_KEY=$GEMINI_API_KEY,ADMIN_API_KEY=$ADMIN_API_KEY" \
  --min-instances 0 \
  --max-instances 1

# --min-instances 0 is what makes this free: nothing runs while nobody is looking,
# and the bill is for the seconds a recruiter is actually on the page. The cost is a
# cold start on the first visit after a quiet spell - the models are baked into the
# image, so it is the image pull and the kb/ ingest, not a download from HuggingFace.
#
# --max-instances 1 is not a cost control, it is a correctness one. Redis lives
# inside the container, so conversation history lives in *that* instance. A second
# instance would answer follow-ups with no memory of the conversation, and it would
# do it intermittently, which is the worst way to find out.
#
# --memory 2Gi covers torch, two MiniLM models, Qdrant, Redis, and the tmpfs the index
# sits in. It idles well under that; the peak is model load at startup.
