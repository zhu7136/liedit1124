# HU-D04 kneel candidate — fresh standard-task delivery (2026-11-26)

This directory is the ONLY entry for this trial. The historical root launch.py,
README, source-overlay, tasks, checkpoints and authorizations do not describe it
and MUST NOT be executed for this trial. Main is not modified.

## Current scope

- User approved public delivery to a NEW branch in zhu7136/liedit1124.
- Requested training: one bounded hour; a paid launch is a separate recorded step.
- Fresh `LimX-HU-D04-01-Flat-BeyondMimic` (NOT No-State-Estimation), seed42,
  4096 parallel environments, 160 actor observations, 29 actions, one4090D24GB.
- Original supplied package/model USD bytes, observations, rewards, physics,
  termination conditions and randomization are preserved. No old checkpoint load.
- Episode time limit explicitly extended from10s to30s for the26.68s motion.
- Headless visual terrain material/debug markers disabled; no physical material
  change. Checkpoint interval100 updates. Max100000 iterations is a safety ceiling,
  NOT the hour limit.
- Worker3600s includes source restoration/initialization/save; PPO soft3420s,
  child cutoff3540s, initialization watchdog600s; no retry or environment fallback.
  Platform allocation/clone/release can extend billing beyond worker runtime.
- Image BJX00000178/V000446, IsaacSim5.1/Lab2.3.2/Python3.11/Torch2.7.0,
  RSL-RL3.1.2 and integrity-pinned offline supplementary wheels.

## Data and delivery

Motion artifact: `zhul136-university-of-science-and-technology-of-china/motion-edits/kneel-hold-reference-candidate:v0`.

Motion SHA256: `12ceddb49bf5e8a795dc69f8132853d86655ec7126524cea6d166be0714c3714`.
1334frames50Hz, original frames0..333 exactly unchanged; transition334..383,
static hold384..1333. Quaternion convention wxyz. All42 recorded body frames
retained. Runtime verifies tracked body indexes/joint ordering against model.

Reuses the16 existing local Git source shards, selecting verified model/source/
vendor/wheels ONLY. A small separately-hashed overlay restores the supplied
baseline source, adds this motion and new worker. No historical worker, motion,
reward injection or checkpoint is copied into the executable workload.

CPU reconstruction, source/asset identity, wheel hashes, prefix preservation,
soft checkpoint stop, NaN rejection and no-resume tests are in cpu_verification.json.
CPU checks are NOT a GPU/Isaac simulation test. No service credentials are stored
in this directory or forwarded by it. No online pip/network fallback.

## Limitations and outputs

The motion is an UNVALIDATED kinematic candidate: knees remain around12mm above
floor and left toe around8mm. It is NOT a proven four-contact pose, stable hold,
self-collision-free motion or hardware-safe command. Default undesired-contact
reward still penalizes knee contact; no new support reward is silently added.

Worker emits update metrics and native logs to outputs/kneel_training, saves a
native final_checkpoint.pt with hashes/configs/completion metadata, and attempts
JIT/ONNX export. Export failures are explicitly reported; a checkpoint alone is
not successful tracking. TensorBoard/offline files use native Flux archival;
artifact retrieval/persistence must be verified after the actual run. No W&B
API key is sent to Flux, so live W&B training curves are not promised.

Original in-file copyright/license notices remain intact. Existing RSL-RL and
wheel distributions retain upstream licenses. No new robot-asset license is asserted.