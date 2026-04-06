# ERC-3 EarthRover Challenge — Outdoor GPS Navigation Fork

> **Fork of [vivekmattam02/ERC-3-earthrover-challenge](https://github.com/vivekmattam02/ERC-3-earthrover-challenge).** The upstream baseline is the indoor-navigation workspace for the NYU EarthRover Challenge. This fork adds a complete **outdoor GPS waypoint navigation runtime** with **monocular depth-based obstacle avoidance** on top of it.

## What this fork adds

All additions are isolated under `T_*` filenames so they can be reviewed independently of the upstream baseline.

| File | What it does |
|---|---|
| **`T_outdoor_runtime.py`** | Top-level outdoor mission runner. Pulls GPS + heading from the SDK, sequences mission checkpoints, optionally arms depth-based safety, and supports `--dry-run`, `--send-control`, `--depth-safety`, and `--auto-mission` flags. |
| **`T_live_depth_viewer.py`** | Real-time Depth-Anything-V2 depth viewer with auto-engaged obstacle avoidance — useful for tuning and field debugging. |
| **`src/T_gps_navigator.py`** | Pure-Python GPS math (no `utm` dependency): haversine distance/bearing, heading-error computation, and a `WaypointManager` that advances through mission checkpoints. |
| **`src/T_outdoor_controller.py`** | Proportional heading controller that steers toward the active GPS waypoint, with an optional monocular-depth obstacle-avoidance overlay. |
| **`src/depth_estimator.py`** | Thin wrapper around Depth-Anything-V2 inference. |
| **`src/depth_safety.py`** | Depth-based safety layer that vetoes / overrides controller commands when obstacles are too close. |
| **RTM client fix** | Patched to send commands to the correct `BOT_UID`. |

### Outdoor stack at a glance

```
GPS + heading (SDK)            ┐
WaypointManager (haversine)    ├──►  T_outdoor_controller (P-controller)  ──►  motor cmds
Mission checkpoint API         ┘                    ▲
                                                    │ veto / slowdown
                          DA-V2 depth  ──►  depth_safety
```

### Running it

```bash
# Dry run, no motor commands
python T_outdoor_runtime.py

# Live with motor commands
python T_outdoor_runtime.py --send-control

# Live with depth-based obstacle avoidance
python T_outdoor_runtime.py --send-control --depth-safety

# Start the SDK mission automatically on launch
python T_outdoor_runtime.py --send-control --auto-mission
```

---

## Upstream baseline (unchanged)

Standalone indoor-navigation workspace for the NYU EarthRover Challenge effort.

This directory is meant to be pushed, cloned, and worked on as its own project. It should not depend on files outside this directory.

## What Is In Here

- `earth-rovers-sdk/`: FrodoBots / EarthRover browser + FastAPI bridge.
- `mbra_repo/`: MBRA / LogoNav research code. MBRA is the relabeling expert from the paper; LogoNav is the deployed-policy side of that codebase.
- `src/`: shared runtime modules copied for this project, including rover interface and safety helpers.
- `third_party/Depth-Anything-V2/`: vendored depth-estimation dependency for optional runtime safety.
- `models/`: local checkpoint location for project-specific model files.

## Local Setup

### 1. Clone the repo and enter this directory

```bash
cd ERC-3-earthrover-challenge
```

### 2. Create your local SDK config

Do not commit a real `.env` file. Each teammate must create their own local copy:

```bash
cp earth-rovers-sdk/.env.sample earth-rovers-sdk/.env
```

Then edit `earth-rovers-sdk/.env` with your actual SDK values:

- `SDK_API_TOKEN`
- `BOT_SLUG`
- `CHROME_EXECUTABLE_PATH`
- `MAP_ZOOM_LEVEL`
- `MISSION_SLUG`

The committed file is only the template: `earth-rovers-sdk/.env.sample`

The local file is ignored by git: `earth-rovers-sdk/.env`

### 3. Place model weights locally

This repo contains code and weight directories, but not the actual large checkpoints.

Expected locations:

- MBRA / LogoNav experiment and deployment weights:
  - `mbra_repo/deployment/model_weights/`
- Project-level checkpoints:
  - `models/`
- Optional depth checkpoints for safety:
  - `third_party/Depth-Anything-V2/checkpoints/`

### 4. Verify the workspace

Run:

```bash
python3 verify_workspace.py
```

The script checks:

- required folders and files exist
- the local SDK `.env` exists and is not still using placeholder values
- runtime weight directories exist and contain checkpoints
- no hardcoded parent-repo paths remain in important project files

If the script exits nonzero, the setup is not ready yet.

## Path Discipline

This workspace should be treated as a standalone project.

- New code should use paths relative to this directory.
- Do not add imports or runtime dependencies that reach back into the parent repo.
- Keep secrets out of git.
- Keep large checkpoints local or in approved artifact storage, not committed directly.

## Current Intended Stack

- Global navigation: topological graph + visual localization
- Local control: a validated online short-horizon controller, with MBRA / LogoNav as the main research reference
- Safety: conservative override layer, optionally with depth and pedestrian checks
- Goal completion: separate image-based checkpoint verification

## Important Notes

- `mbra_repo/` here is a copied workspace version, not the original root copy.
- The local `.env` is intentionally not committed.
- Training configs inside `mbra_repo/train/config/` still contain dataset placeholders; that is expected.
- `verify_workspace.py` is the first check teammates should run after cloning.
