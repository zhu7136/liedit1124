# Fresh NoSE154 kneeling: all self collisions OFF

User authorized 2026-11-27T07:32:49Z. One4090D24GB,4096envs,seed42. Fresh training, no checkpoint init/resume. Motion zhul136-university-of-science-and-technology-of-china/motion-edits/kneel-hold-reference-candidate:v0 SHA256 12ceddb49bf5e8a795dc69f8132853d86655ec7126524cea6d166be0714c3714.

GPU worker limit5400s including init/save, soft5220, child5340, startup600. Zero retries/continuations. Baseline rewards, PD, observations, motion and episode30s unchanged. Disables ALL self collisions BEFORE environment construction and verifies USD attribute on env0 articulation before PPO. Ground contact stays enabled. Motion not dynamics/hardware validated. Separate additive branch; main and previous policy files untouched.
