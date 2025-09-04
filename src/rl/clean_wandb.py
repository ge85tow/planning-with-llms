import wandb

ENTITY = "reginaphoebelisa-technical-university-of-munich"
PROJECT = "GRPO-reboost"

wandb.login(key="76161834310fa3386c0a678d4e30e18138446786")
api = wandb.Api()
FULL_PROJECT_PATH=f"{ENTITY}/{PROJECT}"

# --- Delete runs ---
runs = api.runs(f"{ENTITY}/{PROJECT}")
print(f"Found {len(runs)} runs in project '{PROJECT}'. Deleting...")
for run in runs:
    print(f"Deleting run: {run.name} ({run.id})")
    run.delete()
print("✅ All runs deleted.")

# --- Delete artifacts ---
artifacts = api.artifacts(type=None)  # type=None gives all types
for artifact in artifacts:
    if artifact.project == FULL_PROJECT_PATH:
        print(f"Deleting: {artifact.name} | Type: {artifact.type} | Project: {artifact.project}")
        artifact.delete()
print("✅ All artifacts deleted for this project.")


# --- Delete sweeps ---
try:
    for sweep in api.sweeps(f"{ENTITY}/{PROJECT}"):
        print(f"Deleting sweep: {sweep.id}")
        sweep.delete()
    print("✅ All sweeps deleted.")
except Exception as e:
    print(f"⚠️ Sweep deletion failed: {e}")
