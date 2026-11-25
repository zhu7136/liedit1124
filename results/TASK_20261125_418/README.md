# HU_D04 TASK_20261125_418 policy delivery

Completed2094 PPO updates; saved iteration2093, training3385.590s, time-budget stop, child exit0.
Motion: requested upright-knee-toe candidate,1187frames50Hz. Rewards: original TASK_20261124_300, no added upright/contact reward or knee exemption. Actor154 -> action29.

Unzip this archive in the root of your configured limx_rl_lab/IsaacLab project (the folder containing scripts/rsl_rl/play.py). Use the same code/API as source commit d45241d1b2193169bd96c69e62ccce301e414d6b. That public repository packages its source in source-parts + overlay; --verify-only on launch.py reconstructs verified-source/workload if you need to inspect the matching sources. The scripts assume IsaacLab/IsaacSim and limx_rl_lab are installed.

## Play the LEARNED policy, not only replay the reference

```bash
python scripts/rsl_rl/play.py \
+  --task LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation \
+  --checkpoint "$(pwd)/hud04_task418/final_checkpoint.pt" \
+  --motion_file "$(pwd)/hud04_task418/motion.npz" \
+  --num_envs 1 --device cuda:0 --real-time
```

Run in your IsaacLab Python environment with a display. For headless video add --headless --video --video_length 1250; the script enables cameras for video. replay_npz.py replays reference poses and does NOT evaluate this trained policy. deploy.yaml is controller metadata; play.py loads final_checkpoint.pt and does not accept --deploy_yaml.

final_checkpoint.pt: full RSL-RL checkpoint for play/resume (model+Adam); policy.pt: TorchScript export; policy.onnx: ONNX export; deploy.yaml: clean controller config; motion.npz: exact training target. deploy_native.yaml has Isaac-specific Python tags; never use unsafe YAML loading.

Training completed but convergence, sim-to-sim and hardware deployment are NOT verified. hardware_approved=false. No credentials/private URLs/raw logs.
