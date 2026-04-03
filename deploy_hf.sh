#!/usr/bin/env bash
# Deploy Ticket Triage environment to Hugging Face Spaces
#
# Usage:
#   export HF_TOKEN="hf_your_token_here"
#   bash deploy_hf.sh <your-hf-username>
#
# Example:
#   bash deploy_hf.sh boppanapavanprasad

set -euo pipefail

HF_USER="${1:?Please provide your HF username as the first argument}"
SPACE_NAME="ticket-triage-openenv"
REPO_ID="${HF_USER}/${SPACE_NAME}"

echo "=== Deploying to HF Space: ${REPO_ID} ==="

# Check HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Please set HF_TOKEN environment variable"
    echo "  export HF_TOKEN=\"hf_your_token_here\""
    exit 1
fi

# Login to HF
echo "Logging in to Hugging Face..."
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential

# Create the Space (if it doesn't exist)
echo "Creating HF Space (Docker SDK)..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo(
        repo_id='${REPO_ID}',
        repo_type='space',
        space_sdk='docker',
        private=False,
        exist_ok=True,
    )
    print('Space created/exists: ${REPO_ID}')
except Exception as e:
    print(f'Note: {e}')
"

# Clone the space, copy files, push
WORK_DIR=$(mktemp -d)
echo "Working in: ${WORK_DIR}"

cd "${WORK_DIR}"
git clone "https://huggingface.co/spaces/${REPO_ID}" space_repo 2>/dev/null || {
    git clone "https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${REPO_ID}" space_repo
}

# Copy all project files
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='uv.lock' --exclude='outputs' --exclude='.venv' \
    --exclude='deploy_hf.sh' \
    "${PROJ_DIR}/" "${WORK_DIR}/space_repo/"

cd "${WORK_DIR}/space_repo"
git add -A
git commit -m "Deploy ticket-triage OpenEnv environment" --allow-empty
git push

echo ""
echo "=== Deployment complete! ==="
echo "Space URL: https://huggingface.co/spaces/${REPO_ID}"
echo ""
echo "Wait a few minutes for the Space to build, then test:"
echo "  curl https://${HF_USER}-${SPACE_NAME}.hf.space/health"

# Cleanup
rm -rf "${WORK_DIR}"
