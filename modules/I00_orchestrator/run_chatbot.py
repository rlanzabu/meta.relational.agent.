import subprocess
import os
from pathlib import Path

ORCH_DIR = Path(__file__).resolve().parent
MODULES_DIR = ORCH_DIR.parent
PYTHON = "python3"

# Set up environment with PYTHONPATH
env = os.environ.copy()
env["PYTHONPATH"] = str(MODULES_DIR)

print("🚀 Starting chatbot...")

# Step 1: Run main_docs.py to process documents with explicit parameters
print("📄 Processing documents...")
# Add --chunk-size and --chunk-overlap to see if they're used
result = subprocess.run(
    [PYTHON, "main_docs.py", "--reset", "--chunk-size", "800", "--chunk-overlap", "80"],
    cwd=ORCH_DIR,
    env=env
)
if result.returncode != 0:
    print("❌ Failed to process documents")
    exit(1)

# Step 2: Ask user for query
query = input("💬 What would you like to ask? ")

# Step 3: Run main_query.py with the user's query
print("🤖 Generating answer...")
result = subprocess.run([PYTHON, "main_query.py", query], cwd=ORCH_DIR, env=env)
if result.returncode != 0:
    print("❌ Failed to generate answer")
    exit(1)

print("✅ Done!")