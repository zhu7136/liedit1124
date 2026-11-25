# HU_D04 replacement motion: upright, laterally centered final hold

Authorized ONE fresh GPU trial2026-11-25T13:33:08Z; final-hold-only scope confirmed13:53:46Z.
Source motion_1_333_hold.npz SHA256 c6302553ea3fa923013c0b36618bde25f0bcb5d6da2f58cedb802a8bd4c02393.
Corrected motion SHA256132c760feee8b39f8ca9fd9285e84d89af197bdf4f4a1e7dce42c9ee8a16f798.
1334frames/50fps/26.68s: frames0..233 unchanged,234..333 smooth transition,334..1333 original20s hold.
Torso is waist_pitch_link +Z vertical; whole-body mass COM centered LATERALLY between ankle frames.
No new fore-aft COM target, knee/toe contact target or extra hold extension. Feet endpoint poses retained.
Torso tilt14.2488deg->~0; lateral COM-0.01953266m->~0. All-frame joint-limit check passed.
These are KINEMATIC results, not dynamics, self-collision, convergence or hardware-safety approval.

Training:1x4090D24GB,4096envs,seed42,FRESH no old checkpoint,154actor inputs->29actions,
24steps/update,episode30s(to cover26.68s),worker<=3600s INCLUDING reconstruction/init/export,
soft3420/child3540/startup420 seconds. NO automatic retry/fallback/hardware commands.
Original TASK_20261124_300 tracking rewards/contact penalty unchanged: NO added upright/four-support
reward and NO knee-contact exemption. Legacy hold_support.py/old motion remain INACTIVE inside base archive.
Runtime audit requires new SHA,1334frames,episode30,4096envs,154->29 and original rewards before PPO.

16 local Git shards +10-file SHA256-pinned overlay reconstruct118 source files;23 offline hashed wheels.
No worker HTTP downloads or service credentials.58 CPU checks plus anonymous clone/reconstruction required.
This NEW independent delivery branch does not replace/delete main or any prior result.
Policy/deploy outputs will be delivered separately only after actual remote completion and byte verification.

Run using pinned source descriptor and loader commit:
python launch.py --expected-manifest-sha256 DESCRIPTOR_SHA256 --expected-loader-commit THIS_COMMIT --train
Inspect only(no GPU): python launch.py --expected-manifest-sha256 DESCRIPTOR_SHA256 --verify-only
