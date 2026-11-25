# TASK_20261125_290 — completed bounded training, NOT hardware validated

Source commit `75dc161b14774c95f62ce353ea47483b37c8fafa`; public upload authorized2026-11-25T03:25:13Z.
Completed51PPOupdates, zero-based last iteration50; stop_reason=worker_soft_deadline.
Training3452.991s, configuredworkerlimit3600s, noautomaticretries.
1x4090D24GB,4096envs,seed42,freshtraining,154actorobservations/29actions.

final_checkpoint.pt is the training checkpoint; policy.pt is TorchScript; policy.onnx is the ONNX export.
deploy.yaml is the cleaned controller config. deploy_native.yaml contains IsaacLab-specific Python tags: inspect as data; never use unsafe YAML constructors on untrusted inputs.
controller_bundle.pt is a data-only PyTorch container of exported policy/config/motion/completion bytes; use weights_only=True when unpacking.
retrieval_verification.json lists actual file lengths/hashes and worker checkpoint hash verification.

**Training completed within its time budget, but convergence, sim-to-sim, and physical robot deployment have NOT been validated. hardware_approved=false.**
No credentials, raw platform logs, or private upload/download URLs are included.
