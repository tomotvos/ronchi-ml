# Agent Instructions

> This document is a stub — to be filled in.

## Sections to be completed

- Project overview and current priority
- Priority order: (1) synthetic holdout baseline, (2) preprocessing stabilization, (3) defect training
- Safety rules: do not start defect training until synthetic holdout issues are resolved
- How to run existing tests (pytest)
- How to run synthetic evaluation (command TBD — see docs/synthetic-evaluation-plan.md)
- How to run inference: python src/infer.py (see README.md and docs/IMPLEMENTATION.md)
- How to run training: python src/train.py (see README.md and docs/IMPLEMENTATION.md)
- Coding style expectations
- Documentation expectations
- When to stop and ask for human judgment (scientific assumptions, convention ambiguities)
- Boundaries: do not modify .py files unless an issue explicitly requests it
