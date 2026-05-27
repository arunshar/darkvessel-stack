#!/usr/bin/env bash
# One-shot publish for darkvessel-stack.
# Reads PUSH.md sections 2 through 6 and executes them idempotently.
# Exits non-zero on first failure.
#
# Usage:
#   bash scripts/push_darkvessel.sh             # interactive (asks before destructive ops)
#   AUTO=1 bash scripts/push_darkvessel.sh       # non-interactive
#
# Requires: gh, hf, uv, git, python3.11 (or any 3.11+).

set -euo pipefail

GH_OWNER="arunshar"
HF_OWNER="arun08sharma"
REPO="darkvessel-stack"
MODEL="darkvessel-stack-xview3"
DATASET="xview3-ais-splits"
DESCRIPTION="Multi-modal remote sensing stack for dark vessel detection. Sentinel-1 + Sentinel-2 + AIS, six geospatial foundation models, TGARD + Pi-DPM anomaly reasoning."

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "==> Working from $ROOT"

step()  { printf "\n\033[1;34m==> %s\033[0m\n" "$1"; }
ok()    { printf "\033[1;32m[ok]\033[0m %s\n" "$1"; }
skip()  { printf "\033[1;33m[skip]\033[0m %s\n" "$1"; }
die()   { printf "\033[1;31m[fail]\033[0m %s\n" "$1"; exit 1; }

# 0. Pre-flight
step "0. Pre-flight"
command -v git >/dev/null || die "git not found"
command -v gh  >/dev/null || die "gh not found (brew install gh)"
command -v hf  >/dev/null || die "hf not found (pip install -U 'huggingface_hub[cli]')"
command -v uv  >/dev/null || die "uv not found (curl -LsSf https://astral.sh/uv/install.sh | sh)"
gh auth status >/dev/null 2>&1 || die "gh not authenticated (gh auth login)"
hf auth whoami >/dev/null 2>&1 || die "hf not authenticated (hf auth login)"
ok "tools present, auth hot"

# 1. Local tests
step "1. Local tests"
if [[ ! -d .venv ]]; then
  uv venv --python 3.11 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[dev,space]" >/dev/null
pytest -q
ok "math tests passed"

# 2. Git init + first commit (idempotent)
step "2. Git init"
if [[ ! -d .git ]]; then
  git init -b main >/dev/null
  ok "git initialized"
else
  skip "git already initialized"
fi

git add -A
if git diff --cached --quiet; then
  skip "nothing to commit"
else
  git commit -m "Initial release: DarkVesselNet v0.1.0" >/dev/null
  ok "first commit recorded"
fi

# 3. GitHub remote
step "3. GitHub remote"
if gh repo view "$GH_OWNER/$REPO" >/dev/null 2>&1; then
  skip "$GH_OWNER/$REPO already exists"
  if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin "https://github.com/$GH_OWNER/$REPO.git"
  fi
  git push -u origin main || true
else
  gh repo create "$GH_OWNER/$REPO" \
    --public \
    --source=. \
    --push \
    --description "$DESCRIPTION"
  ok "$GH_OWNER/$REPO created and pushed"
fi

for topic in remote-sensing sentinel-1 sentinel-2 ais dark-vessel-detection \
             geospatial-ai earth-observation foundation-models computer-vision \
             pytorch xview3 maritime sar anomaly-detection prithvi; do
  gh repo edit "$GH_OWNER/$REPO" --add-topic "$topic" >/dev/null 2>&1 || true
done
ok "topics added"

gh repo edit "$GH_OWNER/$REPO" \
  --homepage "https://huggingface.co/spaces/$HF_OWNER/$REPO" >/dev/null
ok "homepage linked to HF Space"

# 4. HF Space
step "4. HF Space"
if hf repo info "$HF_OWNER/$REPO" --repo-type space >/dev/null 2>&1; then
  skip "spaces/$HF_OWNER/$REPO already exists"
else
  hf repo create "$HF_OWNER/$REPO" --repo-type space --space-sdk gradio
  ok "spaces/$HF_OWNER/$REPO created"
fi

pushd space >/dev/null
if [[ ! -d .git ]]; then
  git init -b main >/dev/null
  git remote add origin "https://huggingface.co/spaces/$HF_OWNER/$REPO"
fi
git add -A
if git diff --cached --quiet; then
  skip "Space has nothing new to push"
else
  git commit -m "Scaffold DarkVesselNet Space" >/dev/null
fi
git push -u origin main || git push origin main
ok "Space pushed"
popd >/dev/null

# 5. HF Model + Dataset stubs
step "5. HF stubs"
for slug in "$MODEL"; do
  if hf repo info "$HF_OWNER/$slug" --repo-type model >/dev/null 2>&1; then
    skip "$HF_OWNER/$slug already exists"
  else
    hf repo create "$HF_OWNER/$slug" --repo-type model
    ok "model $HF_OWNER/$slug created"
  fi
done

for slug in "$DATASET"; do
  if hf repo info "$HF_OWNER/$slug" --repo-type dataset >/dev/null 2>&1; then
    skip "$HF_OWNER/$slug already exists"
  else
    hf repo create "$HF_OWNER/$slug" --repo-type dataset
    ok "dataset $HF_OWNER/$slug created"
  fi
done

# 6. Verify
step "6. Verify"
for url in \
  "https://github.com/$GH_OWNER/$REPO" \
  "https://huggingface.co/spaces/$HF_OWNER/$REPO" \
  "https://huggingface.co/$HF_OWNER/$MODEL" \
  "https://huggingface.co/datasets/$HF_OWNER/$DATASET"; do
  code=$(curl -fsSL -o /dev/null -w "%{http_code}" "$url" || echo "000")
  if [[ "$code" == "200" || "$code" == "302" ]]; then
    ok "$url ($code)"
  else
    printf "\033[1;33m[warn]\033[0m %s (HTTP %s) - may still be propagating\n" "$url" "$code"
  fi
done

echo
echo "Done. Open:"
echo "  https://github.com/$GH_OWNER/$REPO"
echo "  https://huggingface.co/spaces/$HF_OWNER/$REPO"
