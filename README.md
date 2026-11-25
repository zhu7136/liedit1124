# HU_D04 upright torso + knees/toes support — one bounded 60min trial

## CURRENT authorization 2026-11-25T02:12:43Z (supersedes all historical notes)

User approved ONE fresh GPU trial <=3600s including initialization/save/export,
and public upload of this trial's source/reference/output, no credentials.
Single4090D24GB,4096envs, seed42,154actor inputs->29actions,25s episodes.
No automatic retries, additional GPU, checkpoint warm-start or OOM fallback.
Input motions/gui00528900_knee_toe_upright_candidate/motion.npz,1187frames50Hz,
SHA256 d0cb1f807c9ae38820368dff57c666d33b47581d351886e5969b6cb8c332fd5a.
Final20s reference has upright upper torso and bilateral knee/toe geometry
support with static COM centering. Dynamics/contacts/policy tracking unproven.

Reward change: only at motionframe>=186 allow knee contacts (other unwanted
contacts still penalized); hold-only upright torso reward weight.5/std.15rad,
four-link contact reward weight.5/5Nthreshold (knees and ankle-roll bodies).
These contact sensors do NOT prove toe-edge location or equal support loads.
Native motion tracking/safety limits/terminations/physics remain unchanged.
Worker hard3600s, PPO soft3420s, child3540s, startup420s, no automatic retry.

deploy.yaml is converted with BaseLoader+explicit scalar schema BEFORE PPO,
validated for154->29 interface, returned by log chunks AND data-only PT bundle.
Original native deploy YAML is retained separately. Parent repeats metadata
delivery outside Isaac logging after child shutdown. Event logger accepts
name= metadata without argument collision. Outputs include final_checkpoint.pt,
policy.pt,policy.onnx,deploy.yaml,deploy_native.yaml,deploy_verification.json,
controller_bundle.pt,metrics/configs/native logs/ZIP via native Flux archival.
No service credentials are forwarded to the GPU worker.

Simulation only: not proof of stable20s hold, closed-loop success or hardware
safety. Previous branches/main/models are preserved. Below is historical
provenance ONLY and is not this trial's authorization or input configuration.

## Superseding authorization 2026-11-24T08:40:48Z:60-minute replacement

The user requested cancellation of the queued30-minute taskTASK_20261124_294
and ONE new60-minute fresh training trial. The cancelled task never started.
This branch changes ONLY the worker budget/run name from the prior No-State
delivery:3600s worker maximum including startup/save/export,3420s PPO soft stop,
3540s child cutoff;420s startup watchdog unchanged. Same one4090D24GB,4096envs,
seed42,154actor inputs29actions,25s episodes and1187-frame plus20s reference.
No auto retries, checkpoint warm start, extra GPU or environment fallback.
Native Flux output storage only. Old tasks/checkpoints/main stay unchanged.
The historical30-minute/187-frame notes below do not describe this new budget.

## Current authorization 2026-11-24T08:28:28Z

- One fresh trial on one4090D24GB,4096 environments; no checkpoint input.
- Task LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation:154actor inputs,
  29actions. Critic retains its privileged observations during training only.
- Current input is motions/gui00528900_toe_vertical_parallel_plus20s/motion.npz,
  1187frames50FPS, SHA256
  3d4bd08212cd96509af9ca46d4f9554b53004a3446dd4c64159c7555842c5fda.
- Episode limit25s, reference random-start sampling retained. Rewards, physics
  and failure criteria unchanged. No knee-contact validation requirement added.
-1800s WORKER limit includes initialization/save/export, PPO softstop1620s,
  child cutoff1740s, startup watchdog420s, no retry/OOM fallback/extra GPU.
- Native Flux output storage only; no general credentials sent to worker.
- Outputs: final_checkpoint.pt, policy.pt (TorchScript), policy.onnx,
  deploy.yaml, metrics/events, configs. Fresh training uses isolated log name.
- Export shape and metadata checks do NOT establish stable tracking, safe
  walk-to-mimic transition or hardware readiness. Simulation only.

The original delivery notes below describe the prior187-frame baseline only;
the current authorization and runtime checks above supersede that input/scope.

User-authorized public delivery of the necessary source, robot USD and motion.
This branch builds on the isolated 64-environment delivery. It does not modify main or reuse
the old low-climb experiments, G1 limits, weights, rewards or datasets.

## Scope

- Flux project HuD04_WBT, one 4090D24GB, 4096 environments, fresh PPO, seed42.
- Replaces the user-stopped 64-environment task; its checkpoints are retained,
  but are NOT loaded here. GPU OOM stops this attempt with no environment fallback.
- Worker budget1800s **including setup/save**; PPO soft stop1620s from entry;
  child cutoff1740s; startup watchdog420s; zero automatic retries/restarts.
- No20-update short cap:100000-iteration safety ceiling plus time deadlines.
- Image BJX00000178 / V000446 (IsaacSim5.1, IsaacLab2.3.2, Python3.11,
  Torch2.7.0); RSL-RL3.1.2 pinned to upstream tree recorded in dependencies.json.
- Torch/CUDA/NumPy/physical materials/rewards/terminations/randomization unchanged.
  Only remote terrain MDL visualization and debug markers are disabled headlessly;
  checkpoint frequency is100 updates; budget/output/body-order checks are added.
- Training input:187frames at50FPS,29joints,42body frames. SHA256:
  `68e6d580354a86caeed1b94e7869475fb3220c1fa4dc95894057301aed62fb35`.
- Input artifact:
  `zhul136-university-of-science-and-technology-of-china/robot-motion-edits/gui00528900_toe_vertical_parallel_candidate:v0`.

## Limitations

The final pose is a kinematic candidate. Dynamics, stability, toe contact,
full self-collision and tracking quality are NOT established by its endpoint
geometry. This is simulation-only training, not authorization for real hardware.
Completion or a saved checkpoint is not evidence of a successful final squat.

## Execution and outputs

Flux runs `gm-run liedit1124/flux/bootstrap.py` with an exact commit and a private,
single-file signed output PUT. The upload destination is never committed here.
No GitHub token, Flux API key or W&B API key is sent to the worker.
No output models or raw logs are published to GitHub.

Offline wheels supplement only missing image imports in a project-local overlay.
No online pip fallback, no GPU/CPU fallback and no failed-run automatic retries.
Body/joint ordering must match the kinematically audited source before learning.

Private result ZIP contains `final_checkpoint.pt` when training completes,
metrics/events JSONL, TensorBoard/config files, native application log and
completion/parent manifests. Source and output hashes are verified separately.
Model retrieval/upload and physical rollout validation remain runtime checks.

Original application source and robot assets were supplied by the user in the
upup1124 project. Existing in-file copyright notices are preserved. RSL-RL is
vendored under its included upstream BSD-3-Clause license; offline wheels retain
their upstream distribution metadata and licenses. No new license is asserted
over the user's robot assets.
