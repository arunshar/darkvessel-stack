# DarkVesselNet push playbook

Ultra-detailed publish guide for `darkvessel-stack`. Three public surfaces go up:

1. GitHub repo `arunshar/darkvessel-stack` (Apache-2.0).
2. Hugging Face Space `spaces/arun08sharma/darkvessel-stack` (Gradio runtime).
3. Hugging Face Model stub `arun08sharma/darkvessel-stack-xview3` (placeholder for trained checkpoint).
4. (Optional) Hugging Face Dataset stub `arun08sharma/xview3-ais-splits` (placeholder for curated splits).

Estimated wall time: 12 minutes if all auth is hot; 25 minutes if you re-auth.

## 0. The 90-second skim

```bash
# auth (skip if already logged in)
gh auth status
hf auth whoami

# pre-flight tests
cd ~/Desktop/cv-portfolio/darkvessel-stack
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev,space]"
pytest -q

# 1. GitHub
git init -b main
git add -A
git commit -m "Initial release: DarkVesselNet v0.1.0"
gh repo create arunshar/darkvessel-stack --public --source=. --push \
  --description "Multi-modal remote sensing stack for dark vessel detection. Sentinel-1 + Sentinel-2 + AIS, six geospatial foundation models, TGARD + Pi-DPM anomaly reasoning."

# 2. HF Space (push only the space/ subtree)
hf repo create arun08sharma/darkvessel-stack --repo-type space --space-sdk gradio
cd space
git init -b main
git remote add origin https://huggingface.co/spaces/arun08sharma/darkvessel-stack
git add -A
git commit -m "Scaffold DarkVesselNet Space"
git push -u origin main
cd ..

# 3. HF Model + Dataset stubs
hf repo create arun08sharma/darkvessel-stack-xview3 --repo-type model
hf repo create arun08sharma/xview3-ais-splits --repo-type dataset

# 4. Smoke verify
curl -fsSL -o /dev/null -w "github   : %{http_code}\n" https://github.com/arunshar/darkvessel-stack
curl -fsSL -o /dev/null -w "hf space : %{http_code}\n" https://huggingface.co/spaces/arun08sharma/darkvessel-stack
curl -fsSL -o /dev/null -w "hf model : %{http_code}\n" https://huggingface.co/arun08sharma/darkvessel-stack-xview3
```

If you just want to fire, stop here. The rest of the doc explains every line, every failure mode, and the recovery path.

## 1. Prerequisites

### 1.1 Tooling

| Tool | Min version | Check | If missing |
| --- | --- | --- | --- |
| `git` | 2.40 | `git --version` | `brew install git` |
| `gh` | 2.50 | `gh --version` | `brew install gh` |
| `uv` | 0.4 | `uv --version` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `hf` | 0.20 | `hf --version` | `pip install -U "huggingface_hub[cli]"` |
| Python | 3.11 | `python3 --version` | `brew install python@3.11` |
| `curl` | any | `curl --version` | preinstalled on macOS |
| `jq` | any (optional) | `jq --version` | `brew install jq` |

The legacy `huggingface-cli` is deprecated as of 2025. We use `hf` throughout. If a copy-pasted command says `huggingface-cli` it will print a deprecation banner and still work, but prefer `hf`.

### 1.2 Auth

```bash
gh auth status                # expect: Logged in to github.com as arunshar
hf auth whoami                # expect: Username: arun08sharma
```

If either fails:

```bash
gh auth login                 # follow the device-code flow, choose HTTPS + browser auth
hf auth login                 # paste a token with at least "write" scope from https://huggingface.co/settings/tokens
```

The HF token is cached at `~/.cache/huggingface/token`. The `gh` token lives in macOS Keychain (`gh auth status` shows the host + keyring).

### 1.3 Required token scopes

- **GitHub:** `repo`, `read:org`, `workflow` (for Actions). The `gh auth login` browser flow picks these automatically.
- **Hugging Face:** create a token at <https://huggingface.co/settings/tokens> with role `Write`. If you plan to gate the model behind acceptance, use `Read + Write` on a fine-grained token.

### 1.4 Git identity

```bash
git config --global user.name  "Arun Sharma"
git config --global user.email "arunshar@umn.edu"
git config --global init.defaultBranch main
git config --global pull.rebase true
```

For *this repo only* (overrides global if you want a different email on public commits):

```bash
cd ~/Desktop/cv-portfolio/darkvessel-stack
git config user.email "arunshar@umn.edu"
```

### 1.5 GPG / SSH signing (optional but recommended for public repos)

```bash
gh ssh-key add ~/.ssh/id_ed25519.pub --title "MBP-2024"      # if you have an ed25519 key
# or generate one:
ssh-keygen -t ed25519 -C "arunshar@umn.edu" -f ~/.ssh/id_ed25519
gh ssh-key add ~/.ssh/id_ed25519.pub --title "MBP-2024"
```

Sign commits:

```bash
gpg --full-generate-key                                         # pick RSA 4096
gpg --list-secret-keys --keyid-format=long
git config --global user.signingkey <KEYID>
git config --global commit.gpgsign true
gh gpg-key add /tmp/pub.asc                                     # export your pubkey first
```

If you have not set up signing and do not want to now, skip it — the rest of the playbook does not require signed commits.

## 2. Pre-flight: local verification

The repo ships with 13 math tests. Run them before publishing so the GitHub README badges do not lie.

```bash
cd ~/Desktop/cv-portfolio/darkvessel-stack
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev,space]"
pytest -q
```

Expected: `13 passed in <2s`.

If a test fails, **stop**. Do not publish. Open the failure trace, fix, re-run.

### 2.1 Space dry-run

Boot the Gradio app on a free local port and curl it:

```bash
PORT=$(python -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
GRADIO_SERVER_PORT=$PORT python space/app.py &
APP_PID=$!
sleep 6
curl -fsSL -o /tmp/dv_smoke.html "http://127.0.0.1:$PORT" \
  && head -c 500 /tmp/dv_smoke.html | grep -q -i "gradio" \
  && echo "Space dry-run: OK"
kill $APP_PID
```

If `curl` returns non-200 or the HTML does not contain "gradio", inspect `/tmp/dv_smoke.html` and fix before publishing.

### 2.2 Requirements dry-run

```bash
uv pip install --dry-run -r space/requirements.txt
```

Expected: resolver finishes without conflict. If a package fails to resolve, pin the offending version in `space/requirements.txt`.

### 2.3 Final tree review

```bash
git status                                              # should be clean (we have not init'd git yet)
find . -type f -not -path "./.venv/*" -not -path "./__pycache__/*" \
       -not -path "./.pytest_cache/*" -not -name "*.pyc" | sort
```

Expected file count: 24 (or close — the README, pyproject, LICENSE, .gitignore, the four docs, the seven src files, the test file, the two space files).

## 3. GitHub publish

### 3.1 Initialize and first commit

```bash
cd ~/Desktop/cv-portfolio/darkvessel-stack
git init -b main
git add -A
git status                                              # sanity check the staged list
```

Inspect what is staged. Anything in `.venv/`, `__pycache__/`, or `*.pt` should be **excluded** by `.gitignore`. If you see them, fix `.gitignore` and `git rm --cached <path>` before committing.

Make the first commit:

```bash
git commit -m "Initial release: DarkVesselNet v0.1.0

Multi-modal remote sensing stack for dark vessel detection.
Sentinel-1 SAR + Sentinel-2 optical + AIS, six geospatial foundation
models (Prithvi-2, Clay v1, SatMAE++, DOFA, SatlasNet, RemoteCLIP)
unified behind a single GeoBackbone interface, seven canonical
EO heads (detect, segment, classify, change, super-res, forecast,
anomaly), TGARD + Pi-DPM anomaly reasoning. Targets xView3-SAR
leaderboard."
```

### 3.2 Create the remote and push

```bash
gh repo create arunshar/darkvessel-stack \
  --public \
  --source=. \
  --push \
  --description "Multi-modal remote sensing stack for dark vessel detection. Sentinel-1 + Sentinel-2 + AIS, six geospatial foundation models, TGARD + Pi-DPM anomaly reasoning."
```

`--source=.` tells `gh` to use the current directory as the repo source; `--push` pushes the current branch to `origin/main`; `--public` makes it visible. `gh` creates the remote, sets `origin`, and pushes in one shot.

### 3.3 Add topics for discoverability

```bash
gh repo edit arunshar/darkvessel-stack --add-topic remote-sensing
gh repo edit arunshar/darkvessel-stack --add-topic sentinel-1
gh repo edit arunshar/darkvessel-stack --add-topic sentinel-2
gh repo edit arunshar/darkvessel-stack --add-topic ais
gh repo edit arunshar/darkvessel-stack --add-topic dark-vessel-detection
gh repo edit arunshar/darkvessel-stack --add-topic geospatial-ai
gh repo edit arunshar/darkvessel-stack --add-topic earth-observation
gh repo edit arunshar/darkvessel-stack --add-topic foundation-models
gh repo edit arunshar/darkvessel-stack --add-topic computer-vision
gh repo edit arunshar/darkvessel-stack --add-topic pytorch
gh repo edit arunshar/darkvessel-stack --add-topic xview3
gh repo edit arunshar/darkvessel-stack --add-topic maritime
gh repo edit arunshar/darkvessel-stack --add-topic sar
gh repo edit arunshar/darkvessel-stack --add-topic anomaly-detection
gh repo edit arunshar/darkvessel-stack --add-topic prithvi
```

GitHub caps at 20 topics; the 15 above are a safe spread. You can edit later from the repo settings UI.

### 3.4 Set the homepage and description (idempotent)

```bash
gh repo edit arunshar/darkvessel-stack \
  --homepage "https://huggingface.co/spaces/arun08sharma/darkvessel-stack" \
  --description "Multi-modal remote sensing stack for dark vessel detection. Sentinel-1 + Sentinel-2 + AIS, six geospatial foundation models, TGARD + Pi-DPM anomaly reasoning."
```

### 3.5 Verify

```bash
gh repo view arunshar/darkvessel-stack --web        # opens in browser
curl -fsSL -o /dev/null -w "%{http_code}\n" https://github.com/arunshar/darkvessel-stack
```

Expect HTTP 200. Confirm the README renders, the topics show up, the file tree is what you expect.

## 4. Hugging Face Space publish

The HF Space is a separate git remote that lives inside `space/`. Only the `space/` subtree is pushed to HF; the main repo on GitHub has the full source.

### 4.1 Create the Space remote

```bash
hf repo create arun08sharma/darkvessel-stack \
  --repo-type space \
  --space-sdk gradio
```

You will see:

```
You are about to create arun08sharma/darkvessel-stack
Proceed? [Y/n]
```

Answer `Y`. The CLI prints the new URL: `https://huggingface.co/spaces/arun08sharma/darkvessel-stack`.

### 4.2 Push the space/ subtree

```bash
cd ~/Desktop/cv-portfolio/darkvessel-stack/space
git init -b main
git remote add origin https://huggingface.co/spaces/arun08sharma/darkvessel-stack
git add -A
git commit -m "Scaffold DarkVesselNet Space"
git push -u origin main
```

If the push prompts for credentials, paste your HF token as the password (not your account password). User: your HF username. Password: the token. If you have `hf auth login`'d, `huggingface_hub` writes credentials to `~/.git-credentials` (or to a credential helper) so this is automatic.

To force the credential helper:

```bash
git config --global credential.helper osxkeychain
```

### 4.3 The Space README YAML frontmatter

The Space root (`space/README.md`) must start with a YAML block that HF parses for build config. The file already has this preconfigured:

```yaml
---
title: DarkVesselNet
emoji: "🛰"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multi-modal RS stack for dark vessel detection (S1+S2+AIS).
tags: [remote-sensing, sentinel-1, sentinel-2, ais, dark-vessel, ...]
---
```

If you change `sdk_version`, make sure the pinned version exists on HF. The current canonical version (May 2026) is `4.44.1`.

### 4.4 Watch the build

```bash
hf repo info arun08sharma/darkvessel-stack --repo-type space
open https://huggingface.co/spaces/arun08sharma/darkvessel-stack
```

The Space build takes 3 to 5 minutes on free CPU tier. Watch the build logs in the right rail. Common build failures and fixes are in section 7.

### 4.5 Hardware tier (optional)

By default the Space runs on `cpu-basic` (free). To bump to a paid tier:

```bash
hf repo update arun08sharma/darkvessel-stack --repo-type space --hw gpu-t4-small
```

Tiers and current pricing (May 2026):

| Tier | vCPU | RAM | GPU | USD / hr | When to use |
| --- | --- | --- | --- | --- | --- |
| cpu-basic | 2 | 16 GB | - | free | This Space (stub mode) |
| cpu-upgrade | 8 | 32 GB | - | 0.03 | trajprompt full pipeline |
| t4-small | 4 | 15 GB | T4 16 GB | 0.40 | physflow-earth inference |
| a10g-small | 4 | 15 GB | A10G 24 GB | 1.05 | sat-splat-distort |
| a100-large | 12 | 142 GB | A100 80 GB | 4.13 | DarkVesselNet full pipeline |

For the stubbed Space, `cpu-basic` is fine. When you swap in real Prithvi-2 forward passes, upgrade to `t4-small` or higher.

### 4.6 Secrets

If a future iteration of the Space needs the Planetary Computer SAS token, set it as a Space secret:

```bash
hf repo update arun08sharma/darkvessel-stack --repo-type space \
  --secret PLANETARY_COMPUTER_KEY="<your sas token>"
```

Secrets are not visible in build logs and are injected as env vars at runtime.

## 5. Hugging Face Model stub

```bash
hf repo create arun08sharma/darkvessel-stack-xview3 --repo-type model
```

The model README YAML frontmatter (paste this when you push real weights):

```yaml
---
library_name: pytorch
license: apache-2.0
tags:
  - remote-sensing
  - sentinel-1
  - sar
  - dark-vessel
  - xview3
  - prithvi
  - tgard
  - pi-dpm
datasets:
  - arun08sharma/xview3-ais-splits
metrics:
  - f1
  - rmse
# no model-index results block: no checkpoint has been trained and no xView3
# metrics have been measured, so none are reported here (add them only after a
# real benchmark run produces them).
---
```

For now an empty repo is enough so the README badge resolves. Push the populated model card when you have a trained checkpoint:

```bash
mkdir -p /tmp/dvxv && cd /tmp/dvxv
git init -b main
git remote add origin https://huggingface.co/arun08sharma/darkvessel-stack-xview3
# write README.md with the frontmatter above
git add README.md && git commit -m "Initial model card"
git push -u origin main
cd ~/Desktop/cv-portfolio/darkvessel-stack
```

## 6. Hugging Face Dataset stub

```bash
hf repo create arun08sharma/xview3-ais-splits --repo-type dataset
```

Push a populated README when you have curated splits:

```yaml
---
license: apache-2.0
task_categories:
  - object-detection
  - image-classification
language:
  - en
tags:
  - remote-sensing
  - xview3
  - dark-vessel
  - sentinel-1
  - ais
size_categories:
  - 1K<n<10K
---
```

## 7. Troubleshooting

### 7.1 `gh repo create` fails with "name already exists"

```bash
gh repo view arunshar/darkvessel-stack         # confirm what is there
# either delete the old repo:
gh repo delete arunshar/darkvessel-stack --yes
# or pick a new name and re-run section 3.2 with the new slug.
```

### 7.2 `hf repo create` fails with "RepoCreationError 409"

The repo already exists. Either:

```bash
hf repo delete arun08sharma/darkvessel-stack --repo-type space --yes
```

or skip the create step and just push to the existing remote.

### 7.3 `git push` to HF fails with 401 Unauthorized

The cached token is expired or has insufficient scope. Re-login:

```bash
hf auth logout
hf auth login                                  # paste a fresh Write-scope token
```

Then re-push.

### 7.4 Space build fails with `ModuleNotFoundError`

A required dependency is missing from `space/requirements.txt`. Check the build log; add the package; commit + push to trigger a rebuild:

```bash
echo "missing-pkg>=1.2" >> space/requirements.txt
cd space
git add requirements.txt
git commit -m "Add missing-pkg"
git push
```

### 7.5 Space build fails with `gradio.exceptions.ServerError`

`sdk_version` in `space/README.md` does not match what your `app.py` imports. Pin to the latest stable:

```bash
hf repo info arun08sharma/darkvessel-stack --repo-type space   # see what runtime HF is using
# edit space/README.md to set sdk_version: 4.44.1 (or whatever current)
```

### 7.6 Space loads but the Gradio UI is blank

Open the browser DevTools console. Most often the `app_file` value does not match the actual file name. Confirm `app_file: app.py` and that `space/app.py` exists.

### 7.7 GitHub README badges 404

Badges link to HF resources; make sure those repos exist *before* GitHub renders the README cache. If you publish GitHub first and HF later, the badges 404 for a few minutes; HF CDN caches eventually catch up.

To force re-resolution:

```bash
gh api -X POST repos/arunshar/darkvessel-stack/dispatches \
  -f event_type="cache-refresh" 2>/dev/null || true
```

(or just wait 10 minutes)

### 7.8 LFS

This repo has no binaries, so LFS is not needed. If you add `.pt` / `.safetensors` checkpoints later:

```bash
brew install git-lfs
git lfs install
git lfs track "*.pt" "*.safetensors"
git add .gitattributes
```

### 7.9 macOS git credential helper not picking up HF token

```bash
git config --global credential.helper osxkeychain
# force-add the token:
printf "protocol=https\nhost=huggingface.co\nusername=arun08sharma\npassword=$(cat ~/.cache/huggingface/token)\n" | git credential-osxkeychain store
```

### 7.10 Push hangs on macOS

Check whether macOS firewall is blocking outbound connections from `git-remote-https`. Disable temporarily under System Settings -> Network -> Firewall, retry, re-enable.

## 8. Re-deploy flow (later updates)

After the initial push, day-2 updates look like this:

```bash
# 1. Source change in the main repo
cd ~/Desktop/cv-portfolio/darkvessel-stack
# edit code, run tests
pytest -q
git add -A
git commit -m "feat: <description>"
git push

# 2. If you also touched space/, push the Space subtree
cd space
git add -A
git commit -m "Space: <description>"
git push
cd ..
```

If you want a one-command sync from the main repo into the Space subtree (advanced):

```bash
git subtree push --prefix=space https://huggingface.co/spaces/arun08sharma/darkvessel-stack main
```

This is fragile when the two histories diverge; the per-subtree-git approach above is simpler.

## 9. GitHub Actions CI (optional)

Create `.github/workflows/test.yml`:

```yaml
name: tests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install uv
        run: pip install uv
      - name: Install deps
        run: |
          uv venv --python 3.11 .venv
          source .venv/bin/activate
          uv pip install -e ".[dev]"
      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest -q
```

Commit and push; the badge `![tests](https://github.com/arunshar/darkvessel-stack/actions/workflows/test.yml/badge.svg)` becomes available.

## 10. Rollback / nuke

If anything goes wrong and you want a clean slate:

```bash
# GitHub
gh repo delete arunshar/darkvessel-stack --yes

# HF Space
hf repo delete arun08sharma/darkvessel-stack --repo-type space --yes

# HF Model stub
hf repo delete arun08sharma/darkvessel-stack-xview3 --repo-type model --yes

# HF Dataset stub
hf repo delete arun08sharma/xview3-ais-splits --repo-type dataset --yes

# Local
cd ~/Desktop/cv-portfolio/darkvessel-stack
rm -rf .git .venv
rm -rf space/.git
```

Then re-run from section 3.

## 11. One-shot script (advanced)

The repo ships with `scripts/push_darkvessel.sh` that runs sections 2 through 6 end-to-end. Inspect it first:

```bash
less scripts/push_darkvessel.sh
```

Then run:

```bash
bash scripts/push_darkvessel.sh
```

The script is idempotent: if any of the four remote resources already exist, it skips the create and proceeds to push. It exits non-zero on the first error so you can pinpoint where to recover.

## 12. Verification checklist

Open these four URLs after publish and confirm they resolve:

- <https://github.com/arunshar/darkvessel-stack>
- <https://huggingface.co/spaces/arun08sharma/darkvessel-stack>
- <https://huggingface.co/arun08sharma/darkvessel-stack-xview3>
- <https://huggingface.co/datasets/arun08sharma/xview3-ais-splits>

For interview links you would share on LinkedIn or in cover letters, the canonical pair is:

- GitHub: <https://github.com/arunshar/darkvessel-stack>
- HF Space (live demo): <https://huggingface.co/spaces/arun08sharma/darkvessel-stack>

That is everything. Fire when ready.
