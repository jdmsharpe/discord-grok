# Discord Grok Bot - Developer Reference

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"        # or: pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Run the bot
python src/bot.py

# Or with Docker
docker-compose up --build
```

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `BOT_TOKEN` | Yes | Discord bot token |
| `XAI_API_KEY` | Yes | xAI API key for Grok requests |
| `GUILD_IDS` | Yes | Comma-separated Discord guild IDs for slash command registration |
| `XAI_COLLECTION_IDS` | No | Comma-separated collection IDs; enables `collections_search` tool |
| `XAI_MCP_PRESETS_JSON` | No | Inline JSON object of named HTTPS MCP presets for `/grok chat` |
| `XAI_MCP_PRESETS_PATH` | No | Path to a JSON file containing named HTTPS MCP presets |
| `SHOW_COST_EMBEDS` | No | Show token/cost embeds on responses (default: `true`; accepts `true/1/yes`) |
| `XAI_PRICING_PATH` | No | Override the bundled `src/discord_grok/config/pricing.yaml` |
| `LOG_FORMAT` | No | `text` (default) or `json` for structured JSON-lines output |

`validate_required_config()` raises `RuntimeError` at startup for missing/blank `BOT_TOKEN` or `XAI_API_KEY`.

## Slash Commands

All commands are nested under `/grok`:

| Command | Description |
| --- | --- |
| `/grok chat` | Start a conversation with Grok (supports advanced tuning, tools, preset-backed MCP, and file attachments) |
| `/grok image` | Generate or edit an image |
| `/grok video` | Generate a video from a prompt or image |
| `/grok tts` | Convert text to speech audio |
| `/grok check_permissions` | Verify bot permissions in the current channel |

`/grok chat` currently exposes:
- Core inputs: `prompt`, `system_prompt`, `model`, `attachment`
- Model tuning: `max_tokens`, `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `reasoning_effort`, `agent_count`
- Tool toggles: `web_search`, `x_search`, `code_execution`, `collections_search`, `mcp`
- Tool refinements: `x_search_images`, `x_search_videos`, `x_search_date_range`, `web_search_images`

## Supported Entry Points

- Launcher: `python src/bot.py` remains supported and delegates to `discord_grok.bot.main`.
- Cog composition contract:

  ```python
  from discord_grok import GrokCog

  bot.add_cog(GrokCog(bot=bot))
  ```

- `discord_grok.bot.main()` now calls `validate_required_config()` before connecting, so missing or blank `BOT_TOKEN` and `XAI_API_KEY` values fail fast at startup.
- `discord_grok` and `discord_grok.cogs.grok` both use lazy `__getattr__` exports so helper imports do not eagerly pull in Discord-heavy modules. Type-only imports keep `pyright src/` aware of those public exports, including the compatibility `BOT_TOKEN` re-export.

## Package Layout

```text
src/
├── bot.py                           # Thin repo-local launcher
└── discord_grok/
    ├── __init__.py
    ├── bot.py
    ├── logging_setup.py             # Structured logging + request-id ContextVar
    ├── config/
    │   ├── __init__.py
    │   ├── auth.py
    │   ├── mcp.py
    │   ├── pricing.py                # YAML loader exposing MODEL_PRICING_CLASSES, IMAGE_PRICING, etc.
    │   └── pricing.yaml              # Canonical pricing data (override via XAI_PRICING_PATH)
    └── cogs/grok/
        ├── __init__.py
        ├── attachments.py
        ├── chat.py
        ├── client.py
        ├── cog.py
        ├── embeds.py
        ├── image.py
        ├── models.py
        ├── responses.py
        ├── speech.py
        ├── state.py
        ├── tooling.py
        ├── video.py
        └── views.py
tests/
├── conftest.py
├── fixtures.py
├── support.py
└── ...
```

Only `src/bot.py` remains at the repo root; code imports should target `discord_grok...`.

## Testing And Patch Targets

- `pytest` runs with `pythonpath = ["src"]`.
- Shared response payloads now live in `tests/fixtures.py`; do not rely on bare `conftest` imports for data fixtures.
- The test suite is organized into module-aligned files: `test_grok_cog`, `test_grok_chat`, `test_grok_client`, `test_grok_commands`, `test_grok_tooling`, `test_grok_embeds`, `test_grok_responses`, `test_grok_state`, `test_button_view`, `test_config_auth`, `test_lazy_imports`, and `test_util`.
- MCP preset coverage now lives in `tests/test_config_mcp.py`, and documentation assertions live in `tests/test_readme.py`.
- `tests/test_package_import.py` is the package import smoke test, and `tests/support.py` holds shared Grok test helpers.
- New tests and patches should target real owners under `discord_grok...`.
- Examples:
  - `discord_grok.cogs.grok.client.RESPONSES_API_URL`
  - `discord_grok.cogs.grok.client.RETRYABLE_STATUS_CODES`
  - `discord_grok.cogs.grok.tooling.XAI_COLLECTION_IDS`
  - `discord_grok.cogs.grok.views.ButtonView`
  - `discord_grok.config.mcp.XAI_MCP_PRESETS`
  - `discord_grok.config.mcp.resolve_mcp_presets`
  - `discord_grok.cogs.grok.models.McpServerConfig`
- Import `GrokCog` from `discord_grok`; do not reintroduce legacy `xai_api` shim paths.

## Validation Commands

```bash
ruff check src/ tests/
ruff format src/ tests/
pyright src/
pytest -q
```

- The repo pre-commit hook (`.githooks/pre-commit`) runs `ruff format` (auto-applied + re-staged), then `ruff check` (blocking), then `pyright` and `pytest --collect-only` as warning-only smoke tests. Resolves tools from `.venv/bin` or `.venv/Scripts` first, then `PATH`.

## Provider Notes

- Conversation state still preserves `previous_response_id`, `response_id_history`, `prompt_cache_key`, and `grok_conv_id`.
- `collections_search` requires `XAI_COLLECTION_IDS`.
- Raw Responses API behavior, retry/backoff handling, and file upload lifecycle now live primarily in `discord_grok.cogs.grok.client`.
- Chat, image, video, and TTS command bodies are delegated from `discord_grok.cogs.grok.cog` into feature modules.
- Remote MCP is configured per `/grok chat` invocation with comma-separated preset names. Presets are loaded from `XAI_MCP_PRESETS_JSON` and `XAI_MCP_PRESETS_PATH`, validated at config-load time, and then persisted as `mcp_servers` on `ChatCompletionParameters`.
- Each MCP preset supports `url` (required HTTPS), `authorization_env_var` (optional), and `allowed_tools` (optional).
- `resolve_selected_tools()` skips the canonical `mcp` marker and only emits MCP tool payloads from validated `mcp_servers`, preventing duplicate MCP entries.
- MCP is intentionally excluded from the built-in tool dropdown so dropdown changes only affect built-in tools.
- The slash-command surface no longer accepts X handle filters or web domain allow/block lists; only media toggles and `x_search_date_range` remain pre-start search refinements.
- Attachment size limits (from `discord_grok.cogs.grok.attachments`): images are capped at 20 MB (`MAX_IMAGE_SIZE`), other files at 48 MB (`MAX_FILE_SIZE`). Patch these constants when writing upload tests.

## Runtime Conventions (Cross-Project)

- **Pricing** is loaded from `src/discord_grok/config/pricing.yaml` by `config/pricing.py` at import time. Chat pricing uses a class indirection (`MODEL_PRICING_CLASSES`) joined to `CHAT_MODEL_CATALOG` in `command_options.py`. Override the YAML via `XAI_PRICING_PATH`. Cross-referenced against [genai-prices/x_ai.yml](https://github.com/pydantic/genai-prices/blob/main/prices/providers/x_ai.yml).
- **Retry**: `client.post_with_retries` wraps every xAI HTTP call with exponential backoff + jitter. `MAX_API_ATTEMPTS=5`, honors `Retry-After` on 429, retries on `RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}` and transport errors.
- **Conversation TTL**: `prune_runtime_state` in `cogs/grok/state.py` evicts conversations older than `CONVERSATION_TTL` (12h) every 15 minutes via `@tasks.loop`. Caps at `MAX_ACTIVE_CONVERSATIONS`. Daily costs retained for `DAILY_COST_RETENTION_DAYS` (30).
- **Request IDs**: `cog_before_invoke` (and `on_message`) bind a fresh 8-char hex id via `discord_grok.logging_setup.bind_request_id`. All downstream `logger.info`/`warning`/`error` calls automatically include the id. Set `LOG_FORMAT=json` for JSON-lines output.
