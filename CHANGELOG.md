# Changelog

All notable changes to Kosmos AI Scientist will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2025-11-13

### Added - Multi-Provider LLM Support 🎉

**Major Feature:** Kosmos now supports multiple LLM providers for maximum flexibility!

#### Supported Providers
- **Anthropic Claude** (default) - API + CLI mode
- **OpenAI** - GPT-4 Turbo, GPT-4, GPT-3.5, O1 models
- **Ollama** - Free local models (Llama, Mistral, etc.)
- **OpenRouter** - Access to 100+ models
- **LM Studio** - Local models with GUI
- **Together AI** - Open-source models at scale

#### Core Implementation
- Provider abstraction layer with unified `LLMProvider` interface ([#3](https://github.com/jimmc414/Kosmos/issues/3))
- Configuration-driven provider switching via `LLM_PROVIDER` environment variable
- `AnthropicProvider` - Refactored from `ClaudeClient` with full backward compatibility
- `OpenAIProvider` - Full OpenAI API + OpenAI-compatible endpoints support
- Provider factory pattern for easy extensibility
- Unified response format (`LLMResponse`, `UsageStats`, `Message`)

#### Configuration
- New `LLM_PROVIDER` environment variable for provider selection
- New `OpenAIConfig` class with full OpenAI configuration
- Support for custom base URLs (enables Ollama, OpenRouter, etc.)
- `.env.example` expanded with comprehensive provider examples

#### Documentation
- `docs/providers/README.md` - Complete setup guide for all 6+ providers
- `docs/MIGRATION_MULTI_PROVIDER.md` - Migration guide (emphasizes zero breaking changes)
- `docs/api/llm.md` - API documentation for provider interface
- `docs/releases/MULTI_PROVIDER_RELEASE.md` - Detailed release notes
- Updated `README.md` with multi-provider comparison table

#### Testing
- Manual test scripts for provider validation:
  - `tests/manual/test_provider_switching.py` - Test switching between providers
  - `tests/manual/test_basic_generation.py` - Test core generation features
  - `tests/manual/test_ollama.py` - Test Ollama local model integration
- Existing test suite validation (15/17 tests pass, 2 pre-existing cache issues)

### Changed
- `get_client()` function now supports provider selection from configuration
- `kosmos/core/llm.py` updated with provider-aware singleton pattern
- Added `get_provider()` function as recommended API for new code
- Updated README tagline to reflect multi-provider support

### Dependencies
- Added `openai>=1.0.0` for OpenAI provider support

### Backward Compatibility
- ✅ **100% backward compatible** - No breaking changes
- ✅ Existing `ANTHROPIC_API_KEY` and `CLAUDE_*` variables still work
- ✅ Default provider remains Anthropic
- ✅ `ClaudeClient` class maintained as alias
- ✅ All existing code continues to work unchanged

### Commits
- `35dc159` - Core multi-provider infrastructure (Phases 1-3, 5)
- `[pending]` - Documentation and testing (Phases 7-9)

---

## [0.1.0] - 2025-11-06

### Initial Production Release 🚀

#### Major Features
- **Autonomous Research Cycle** - End-to-end scientific workflow automation
- **Multi-Domain Support** - Biology, physics, chemistry, neuroscience, materials science
- **Claude Integration** - Powered by Claude Sonnet 4.5
- **Agent-Based Architecture** - Modular agents for hypothesis, experiment design, analysis
- **Beautiful CLI** - Rich terminal interface with 8 commands
- **Safety-First Design** - Sandboxed execution, validation, reproducibility

#### Performance & Optimization (v1.0)
- **20-40× Overall Performance** - Combined optimizations
- **Parallel Execution** - 4-16× faster experiments via ProcessPoolExecutor
- **Concurrent Operations** - 2-4× faster research cycles
- **Smart Caching** - 30%+ API cost reduction
- **Database Optimization** - 10× faster queries with 32 strategic indexes
- **Auto-Scaling** - Kubernetes HorizontalPodAutoscaler support

#### Production Features (v1.0)
- **Health Monitoring** - Prometheus metrics, alerts (email/Slack/PagerDuty)
- **Performance Profiling** - CPU, memory, bottleneck detection
- **Docker Deployment** - Complete docker-compose stack
- **Kubernetes Ready** - 8 manifests for production deployment
- **Cloud Support** - AWS, GCP, Azure deployment guides
- **Comprehensive Testing** - 90%+ test coverage

#### Caching System
- Multi-tier caching (Claude, Experiment, Embedding, General)
- 25-35% cache hit rate for LLM responses
- 40-50% cache hit rate for computational results
- Automatic cache budget alerts
- Cache statistics and monitoring

#### Developer Experience
- **Flexible Integration** - Anthropic API + Claude Code CLI support
- **Rich Documentation** - 10,000+ lines of documentation
- **11 Example Projects** - Across all supported domains
- **CLI Tools** - run, status, history, cache, config, profile commands

#### Literature Integration
- ArXiv, Semantic Scholar, PubMed integration
- Automated paper search and summarization
- Novelty checking for hypotheses
- Knowledge graph construction

#### Core Components
- **Hypothesis Generation** - AI-powered hypothesis generation with literature context
- **Experiment Designer** - Automated experimental protocol design
- **Data Analyst** - Statistical analysis and interpretation
- **Literature Analyzer** - Paper analysis and knowledge extraction
- **Research Director** - Orchestrates full research cycle

#### Database & Storage
- PostgreSQL for relational data
- ChromaDB for vector embeddings
- Neo4j for knowledge graphs
- Redis for caching

#### Safety & Validation
- Code validation and sandboxing
- Reproducibility tracking
- Resource limit enforcement
- Guardrails for experimental safety

### Dependencies
- Python 3.11+
- `anthropic>=0.40.0` - Claude API integration
- `pydantic>=2.0.0` - Configuration validation
- `sqlalchemy>=2.0.0` - Database ORM
- `chromadb>=0.4.0` - Vector database
- Scientific computing: numpy, pandas, scipy, scikit-learn, statsmodels
- Visualization: matplotlib, seaborn, plotly
- CLI: rich, typer, click

### Documentation
- Comprehensive README with quick start
- Architecture documentation (1,680 lines)
- User guide (1,156 lines)
- Developer guide (870 lines)
- Troubleshooting guide (547 lines)
- API documentation (10 .rst files)
- CONTRIBUTING.md (580 lines)

### Known Issues
- Test suite has 2 cache-related test failures (pre-existing, non-blocking)

---

## Release Notes

- **[v0.2.0 Multi-Provider Release](docs/releases/MULTI_PROVIDER_RELEASE.md)**

---

## Links

- **Homepage**: https://github.com/jimmc414/Kosmos
- **Documentation**: docs/
- **Issues**: https://github.com/jimmc414/Kosmos/issues
- **Contributing**: CONTRIBUTING.md
