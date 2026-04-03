#!/usr/bin/env python3
"""
Deploy Ticket Triage environment to Hugging Face Spaces.

Usage:
    export HF_TOKEN="hf_your_token_here"
    python deploy_to_hf.py --username YOUR_HF_USERNAME

Requires: pip install huggingface_hub
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Deploy Ticket Triage to HF Spaces")
    parser.add_argument("--username", required=True, help="Your HF username")
    parser.add_argument("--space-name", default="ticket-triage-openenv", help="Space name")
    parser.add_argument("--private", action="store_true", help="Make space private")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set")
        print("  export HF_TOKEN='hf_xxxxx'")
        print("  Get a token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    repo_id = f"{args.username}/{args.space_name}"
    project_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"=== Deploying to HF Space: {repo_id} ===")
    print(f"Project dir: {project_dir}")

    # Login
    login(token=token)
    api = HfApi()

    # Verify auth
    user_info = api.whoami()
    print(f"Logged in as: {user_info['name']}")

    # Create the Space
    print(f"Creating Space: {repo_id} ...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        print(f"Space created/exists: {repo_id}")
    except Exception as e:
        print(f"Warning creating space: {e}")

    # Upload all project files
    print("Uploading files...")
    
    # Files to upload
    files_to_upload = [
        "Dockerfile",
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "openenv.yaml",
        "inference.py",
        ".dockerignore",
        ".gitignore",
        "server/__init__.py",
        "server/models.py",
        "server/app.py",
    ]

    for filepath in files_to_upload:
        full_path = os.path.join(project_dir, filepath)
        if os.path.exists(full_path):
            print(f"  Uploading: {filepath}")
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=filepath,
                repo_id=repo_id,
                repo_type="space",
            )
        else:
            print(f"  SKIP (not found): {filepath}")

    print(f"\n=== Deployment complete! ===")
    print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
    print(f"API URL:   https://{args.username}-{args.space_name}.hf.space")
    print(f"\nWait 2-3 minutes for the Space to build, then test:")
    print(f"  curl https://{args.username}-{args.space_name}.hf.space/health")
    print(f"  curl -X POST https://{args.username}-{args.space_name}.hf.space/reset \\")
    print(f'    -H "Content-Type: application/json" -d \'{{"task_id": "task_easy"}}\'')


if __name__ == "__main__":
    main()
