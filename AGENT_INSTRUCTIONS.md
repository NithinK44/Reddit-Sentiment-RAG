# Agent Instructions

This file contains specific constraints and rules for the AI agent working on this project.

## Rules

1. **Test Failures**: Do not attempt to fix terminal test failures more than twice in a single run.
2. **Verification Failures**: If a code diff fails to compile or pass verification twice consecutively, STOP immediately and ask the user for guidance.
3. **Refactoring**: Confirm with the user before executing any multi-file refactors.
