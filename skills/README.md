# Capability teacher skills

Stage 2 generates each expert's skill from its own training trajectories:

```bash
bash scripts/train.sh skill
```

Generated files are stored outside the checkout at
`$OUTPUT_DIR/skill_extraction/skills/capabilityN/SKILL.md`, alongside evidence
ledgers and cited JSON cards. Set `SKILL_WORK_DIR` to override the extraction
output directory, or `TEACHER_SKILL_DIR` to use another set of skills in stage 3.
Flat `capability1.txt` through `capability4.txt` directories are also supported.

MOPD loads the corresponding skill as a suffix in each teacher's user turn when
scoring student-generated responses. Student rollouts and evaluation do not
receive skills.
