# HU-D04 kneel candidate — fresh standard-task delivery (2026-11-26)

## CURRENT EXECUTION OVERRIDE: one setup check, maximum600s

User authorized the proposed single1x4090D24GB setup check at2026-11-26T09:32:27Z.
This revision uses worker600s inclusive of restore/init/save, soft480s,
child cutoff570s and startup watchdog420s, maximum2 PPO iterations, zero retries.
4096envs/seed42, original model, motion, standard task, rewards and physics remain
unchanged. It is plumbing validation, NOT the one-hour training or a tracking
quality result. No automatic continuation, fallback or second paid task is allowed.
Allocation/clone/release billing overhead may exceed worker time; actual platform
billing applies. Previous60min settings/authorization notes below are historical;
this override and delivery.json govern this setup-check revision.

## Fix2: entry-import closure repair; NO new GPU launch authorized

TASK_20261126_468 failed before simulation/PPO due to a missing `hashlib`
import. Fix1 corrected that import, but the delivery omitted the ORIGINAL
`scripts/list_envs.py`, which train.py imports before AppLauncher. Both platform
records TASK_20261126_492 and TASK_20261126_493 ran approximately20s and ended
before PPO. Do not assume the earlier one-additional-attempt approval remains
available: it is exhausted. No further paid launch is authorized by this revision.

Fix2 selects the byte-identical original list_envs.py from existing Git shards
and regenerates BOTH integrity manifests. No robot, source package, reward,
physics, motion, task, environment count or training setting changes versus fix1.

Fresh CPU checks: 9 existing groups in cpu_verification.json plus5 entry/import
groups in entry_closure_verification.json. The actual delivered train.py,
list_envs.py, cli_args.py and motion_args.py execute through task discovery and
argument parsing to an intentionally MOCKED AppLauncher boundary. All9 distinct
absolute project-local imports resolve;5 registered tasks and15 configuration
entry-point symbols exist. External Gym registration, PrettyTable, argcomplete
autocomplete and AppLauncher are MOCKED. This is NOT actual Isaac/CUDA/PPO or
dynamics validation. Older dependency_regression.json is retained historical
fix1 evidence, not a new GPU result. No smoke task or paid task is started here.

Main and both previous delivery branches remain unchanged. Obtain fresh explicit
resource/time/billing approval BEFORE creating or launching another paid attempt.

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