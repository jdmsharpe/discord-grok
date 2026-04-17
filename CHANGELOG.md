# Changelog

## v1.1.0

### feat
- Add `Conversation.touch()` and `updated_at` tracking, and evict conversations older than 12h via `prune_runtime_state()` on a 15-minute `@tasks.loop`; cap active conversations at 100 and retain `daily_costs` for 30 days (fixes a real memory leak).
- Extract chat pricing classes (`premium`, `fast`, `code_fast`, `legacy_premium`, `mini`), `IMAGE_PRICING`, `VIDEO_PRICING_PER_SECOND`, `TTS_PRICING_PER_MILLION_CHARS`, and `TOOL_INVOCATION_PRICING` into `src/discord_grok/config/pricing.yaml`, loaded via `src/discord_grok/config/pricing.py`; preserve the class-based pricing indirection where `CHAT_MODEL_CATALOG` in `command_options.py` owns model-to-class assignment and the YAML owns class-to-price; override at runtime via `XAI_PRICING_PATH`.
- Add structured logging with per-request IDs in `src/discord_grok/logging_setup.py` (`REQUEST_ID` ContextVar, `bind_request_id()`, `configure_logging()`); `cog_before_invoke` and `on_message` bind fresh 8-char hex ids, and `LOG_FORMAT=json` enables JSON-lines output.

### fix
- Correct the `mini` class `cached_input_per_million` price from `0.07` to `0.075` per current xAI docs (cross-referenced against [genai-prices/x_ai.yml](https://github.com/pydantic/genai-prices/blob/main/prices/providers/x_ai.yml)).

### chore
- Bump project version to `1.1.0`.
- Add `PyYAML~=6.0` runtime dependency for the pricing loader.
- Canonical `.githooks/pre-commit`: `ruff format` (auto-applied + re-staged), `ruff check` (blocking), `pyright` (warning-only), `pytest --collect-only` (warning-only smoke); identical across all six `discord-*` repos.

### test
- Add 7 new pricing tests in `tests/test_config_pricing.py` covering YAML load, class indirection, and `XAI_PRICING_PATH` overrides.
- Add 8 new logging tests in `tests/test_logging_setup.py` covering `REQUEST_ID` binding, formatter selection, and JSON-lines output.
- Extend `tests/test_grok_state.py` with TTL prune coverage (`Conversation.touch()`, eviction thresholds, active-conversation cap, daily-cost retention).
- Total test count goes from 225 to 240.

### docs
- Refresh `README.md` with new `XAI_PRICING_PATH` and `LOG_FORMAT` environment variables.
- Update `.claude/CLAUDE.md` with the YAML pricing loader layout, TTL prune conventions, and structured logging notes.

### compare
- [`v1.0.2...v1.1.0`](https://github.com/jdmsharpe/discord-grok/compare/v1.0.2...v1.1.0)
