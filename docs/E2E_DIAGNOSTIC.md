# Kosmos E2E Diagnostic Report

## Summary

**Update:** Individual agents work correctly but are slow due to LLM API latency:
- HypothesisGeneratorAgent: ~19 seconds per call (with literature search)
- ExperimentDesignerAgent: ~89 seconds per call (multiple LLM passes)

The `kosmos run` CLI hangs because ResearchDirectorAgent uses message-passing without an agent runtime.

The `baseline_workflow.py` times out because the default timeout (120s) is shorter than a single experiment design cycle.

## What Works

| Component | Test Method | Result |
|-----------|-------------|--------|
| HypothesisGeneratorAgent | `agent.generate_hypotheses(...)` | 19s, returns valid hypothesis |
| ExperimentDesignerAgent | Unit tests | Passes |
| DataAnalystAgent | Unit tests | Passes |
| LiteratureAnalyzerAgent | Semantic Scholar API | Works unauthenticated |
| ContextCompressor | Smoke test | Passes |
| ArtifactStateManager | Smoke test | Passes |
| PlanCreator/Reviewer | Smoke test | Passes |
| NoveltyDetector | Smoke test | Passes |
| SkillLoader | Smoke test | Passes |
| ResearchWorkflow (async) | Smoke test | Passes (1 cycle) |
| kosmos doctor | CLI | All checks pass |

## What Doesn't Work

| Component | Issue | Root Cause |
|-----------|-------|------------|
| `kosmos run` CLI | Hangs after banner | ResearchDirectorAgent message-passing |
| ResearchDirectorAgent.execute() | Sends messages, nothing happens | No agent runtime processing messages |
| baseline_workflow.py | Timeout | Unknown (needs investigation) |

## Root Cause Analysis

### Message-Passing Architecture Without Runtime

`ResearchDirectorAgent` uses `_send_to_hypothesis_generator()`, `_send_to_experiment_designer()`, etc. These methods:

1. Create `AgentMessage` objects
2. Call `self.send_message(to_agent=target_agent, ...)`
3. Store in `pending_requests`
4. **Wait for response that never comes**

The messages go nowhere because there's no:
- Agent registry with running agent instances
- Message broker/router
- Event loop processing incoming messages

### Evidence

```python
# This works (direct invocation):
agent = HypothesisGeneratorAgent(config={})
result = agent.generate_hypotheses(research_question='...', domain='...')  # Returns in 19s

# This doesn't work (message-passing):
director = ResearchDirectorAgent(research_question='...', domain='...')
director.execute({'action': 'start_research'})  # Returns immediately, nothing happens
```

## Recommended Fix Path

### Phase 1: Direct Agent Invocation (Immediate)

Create a simplified orchestrator that calls agents directly instead of via messages:

```python
class DirectResearchOrchestrator:
    def __init__(self, research_question, domain):
        self.hypothesis_agent = HypothesisGeneratorAgent(config={})
        self.experiment_agent = ExperimentDesignerAgent(config={})
        self.analyst_agent = DataAnalystAgent(config={})

    def run_cycle(self):
        # Step 1: Generate hypotheses (direct call)
        hypotheses = self.hypothesis_agent.generate_hypotheses(
            research_question=self.research_question,
            domain=self.domain
        )

        # Step 2: Design experiments (direct call)
        for h in hypotheses.hypotheses:
            protocol = self.experiment_agent.design_experiment(
                hypothesis=h
            )

        # Step 3: Execute and analyze...
```

This approach:
- Uses existing, working agent implementations
- Bypasses broken message-passing
- Can be done in a few hours

### Phase 2: Fix CLI to Use Direct Orchestrator

Update `kosmos/cli/commands/run.py` to use `DirectResearchOrchestrator` instead of `ResearchDirectorAgent`.

### Phase 3: Add Agent Runtime (Future)

If message-passing is desired:
1. Implement agent registry that spawns/tracks agent processes
2. Add message queue (in-memory or Redis)
3. Implement message routing
4. Add response handlers

## Quick Verification Commands

```bash
# Works - individual agent
python3 -c "
from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent
agent = HypothesisGeneratorAgent(config={})
result = agent.generate_hypotheses(research_question='Does X affect Y?', domain='biology')
print(f'Generated {len(result.hypotheses)} hypotheses')
"

# Doesn't work - CLI
timeout 30 kosmos run "Does X affect Y?" --domain biology --max-iterations 1

# Works - smoke test workflow
python scripts/smoke_test.py
```

## Files to Modify

| Priority | File | Change |
|----------|------|--------|
| P0 | `kosmos/workflow/direct_orchestrator.py` | NEW: Direct agent invocation |
| P0 | `kosmos/cli/commands/run.py` | Use DirectResearchOrchestrator |
| P1 | `scripts/baseline_workflow.py` | Debug timeout issue |
| P2 | `kosmos/agents/research_director.py` | Keep for future message-passing |

## Open Questions

1. Why does `baseline_workflow.py` timeout when direct agent calls work?
2. Is message-passing architecture needed for production?
3. Should we add async/concurrent agent execution?

## Related Issues

- **ISSUE_SKILLLOADER_BROKEN.md** - SkillLoader returns None, blocking Gap 3 (Agent Integration)
  - 116 skill files exist but aren't being loaded
  - COMMON_SKILLS references non-existent files (pandas, numpy, etc.)
  - Domain-to-bundle mapping incomplete
