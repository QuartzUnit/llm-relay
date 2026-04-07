# llm-relay

Unified LLM usage management — API proxy, session diagnostics, multi-CLI orchestration.

## Features

- **Proxy**: Transparent API proxy with cache/token monitoring and 12-strategy pruning
- **Detect**: 8 detectors (orphan, stuck, inflation, synthetic, bloat, cache, resume, microcompact)
- **Recover**: Session recovery and doctor (7 health checks)
- **Guard**: 4-tier threshold daemon
- **Cost**: Per-1% cost calculation and rate-limit header analysis
- **Orch**: Multi-CLI/API orchestration (Claude Code, Gemini CLI, Codex CLI)

## Install

```bash
pip install llm-relay
```

## License

MIT
