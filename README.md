# HU_D04 new motion with task300 original rewards

Authorized 2026-11-25T06:30:23Z. One fresh GPU trial, 4096 environments, seed42, worker ceiling3600s including initialization/export; no automatic retries.

Target: motion_knee_toe_upright_candidate.npz; worker motion.npz has identical bytes, SHA256 d0cb1f807c9ae38820368dff57c666d33b47581d351886e5969b6cb8c332fd5a.

Reward baseline: TASK_20261124_300. All original tracking rewards and undesired-contact penalty retained. NO added upright reward, NO four-support/contact reward, NO knee-contact exemption. hold_support.py remains an INACTIVE legacy file in the base archive and is not imported. Runtime audit rejects either added reward and requires original contact-penalty function.

The 16 pinned local Git shards contain the base source; the 8-file SHA256-pinned overlay restores task300 reward behavior and retains verified offline dependencies/deploy export. No worker HTTP downloads or service credentials. Loader reconstructs and verifies116 source files before execution.

52 CPU integrity/configuration checks passed; this does not prove GPU speed, training convergence or hardware safety. The preceding task290 sampling regression remains a measured risk. No hardware deployment approved.
