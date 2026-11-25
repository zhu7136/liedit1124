# Upright source-delivery repair — STAGED, NOT READY TO SUBMIT

This orphan branch contains only the small loader and pinned source descriptor.
The original main and training branch are unchanged. Do NOT submit this branch
as a fix: Flux can still fall back to the unchanged README-only main before
the loader runs. A separate tiny public repository with this entry on its main
is needed to avoid cloning the original repository history and default fallback.

Publication blocker: current GitHub credential can write this repository but
POST /user/repos returned 403. Create an empty public repository named
hud04-upright-delivery-20261125 and grant the existing credential Contents:
read/write for it. No token value should be pasted into chat or code.

CPU-only verification after obtaining these files:

```bash
python launch.py --expected-manifest-sha256 25a59fda4b7e16c602ff3a55e43820c2717155ceb3bc8d60018f454ddcea07da --verify-only
```

Downloads 16 immutable <=3 MiB source parts from public GitHub blob APIs with
no credentials. Verifies individual SHA256/Git blob hash, full ZIP hash, all
internal file hashes, safe paths and fresh extraction directory. Any failure
aborts; there is no alternate source/default branch in the loader. Only an
explicit --train and exact loader commit allows training. Entry runs bootstrap
in the same gm-run process to preserve platform SDK finalization.

Only bootstrap time accounting changed: source download is included in the
3600-second worker budget. Source delivery startup deadline is 180 seconds;
overall startup deadline remains 420 seconds. Scheduling/platform cloning is
outside this worker budget. Motion, rewards, training interface and deploy
export code are unchanged. CPU verification is not proof of worker egress,
Isaac/CUDA readiness, training success or a newly trained policy/deploy.yaml.

No GPU task created, restarted or started by this repair.
