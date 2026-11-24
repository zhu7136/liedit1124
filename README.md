# HU_D04 toe-down, parallel-shin candidate — bounded resume delivery

User-authorized public delivery of the necessary source, robot USD and motion.
This branch builds on the completed 4096-environment delivery. It does not modify main or reuse
the old low-climb experiments, G1 limits, weights, rewards or datasets.

## Scope

- Flux project HuD04_WBT, one 4090D24GB, 4096 environments, resumed PPO, seed42.
- User approval 2026-11-24T06:54:42Z: one additional 30-minute trial.
- Resume the completed TASK_20261124_187 final_checkpoint.pt (936 completed updates),
  SHA256 84946a9c531e60b38a926874506c855fe95747a955f63cb1ac2b3c1f19a969a5.
- Native RSL-RL load restores actor, critic, action noise and Adam; tensor/state
  equality is checked, adaptive learning rate restored from Adam and iteration
  labels continue at936. Environment and RNG state are not in the source checkpoint
  and reset. This is optimizer continuation, not a bit-identical simulator replay.
- All previous checkpoints are retained. OOM stops with no fallback or retry.
- Inputs mount through Flux checkpoint fields. No model is published in this source
  branch, and no general service keys or signed output URL are sent to the worker.
- Final files remain in native Flux output storage; checkpoint will be retrieved,
  verified and copied to the W&B submission-tracking run after training.
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

Flux runs `gm-run liedit1124/flux/bootstrap.py` with an exact commit and a private
checkpoint mount. Outputs use native Flux archival, not a signed third-party PUT.
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
