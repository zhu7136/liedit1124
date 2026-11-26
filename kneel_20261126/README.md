# HU-D04 kneel — No-State-Estimation,154-input fresh1h trial

## ACTIVE AUTHORIZATION2026-11-26T10:02:08Z

User explicitly requested immediate ONE formal one-hour GPU training attempt.
1x4090D24GB,4096envs,seed42;fresh training with native
`LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation`. Actor154/actions29;
remove actor motion_anchor_pos_b and base_lin_vel, retain original privileged
critic. No resume, no160-input checkpoint initialization, no automatic retry.

Worker hard3600s includes source restoration/init/save; soft3420s, child3540s,
startup watchdog600s. Iteration100000 is a safety ceiling, NOT the time limit;
this is NOT the2-update setup check. Platform allocation/clone/release overhead
can add actual billing time. No new GPU task beyond this single attempt is approved.

Prior task502 validated the STANDARD160-input task only. The154-input profile
has passed source/entry/argument/CPU regressions but awaits its own runtime
observation audit on this trial; fail closed rather than substituting standard.
Original USD/source package/motion/rewards/physics unchanged. Motion remains an
unvalidated kneeling candidate, not a proven stable/hardware-safe four-contact
motion; default knee-contact penalties remain. Do not infer task success from
worker launch, training completion, checkpoint presence or policy export alone.

Prior documentation below is retained as HISTORY; this section and delivery.json
govern the new trial. Main and previous delivery branches must remain untouched.

## CURRENT LOCAL PREPARATION: No-State-Estimation requested2026-11-26T09:54:14Z

The active prepared task is `LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation`.
Expected actor input154, not160: remove motion_anchor_pos_b(3) and base_lin_vel(3).
Original privileged critic remains unchanged for asymmetric actor-critic training.
Worker selection, runtime observation audit, export input and checkpoint metadata
all target154. Original robot/source package/motion/rewards/physics unchanged.
Fresh training only: never resume from the160-input setup checkpoint from task502.
This profile has NOT been GPU tested, published or launched. No new paid launch
is authorized. Earlier standard-task details below describe prior revisions only.

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