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
| `SHOW_COST_EMBEDS` | No | Show token/cost embeds on responses (default: `true`; accepts `true/1/yes`) |

`validate_required_config()` raises `RuntimeError` at startup for missing/blank `BOT_TOKEN` or `XAI_API_KEY`.

## Slash Commands

All commands are nested under `/grok`:

| Command | Description |
| --- | --- |
| `/grok chat` | Start a conversation with Grok (supports tools, MCP, file attachments) |
| `/grok image` | Generate or edit an image |
| `/grok video` | Generate a video from a prompt or image |
| `/grok tts` | Convert text to speech audio |
| `/grok check_permissions` | Verify bot permissions in the current channel |

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
    ├── config/
    │   ├── __init__.py
    │   └── auth.py
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
- `tests/test_package_import.py` is the package import smoke test, and `tests/support.py` holds shared Grok test helpers.
- New tests and patches should target real owners under `discord_grok...`.
- Examples:
  - `discord_grok.cogs.grok.client.RESPONSES_API_URL`
  - `discord_grok.cogs.grok.client.RETRYABLE_STATUS_CODES`
  - `discord_grok.cogs.grok.tooling.XAI_COLLECTION_IDS`
  - `discord_grok.cogs.grok.views.ButtonView`
  - `discord_grok.cogs.grok.tooling.validate_mcp_server_input`
  - `discord_grok.cogs.grok.models.McpServerConfig`
- Import `GrokCog` from `discord_grok`; do not reintroduce legacy `xai_api` shim paths.

## Validation Commands

```bash
ruff check src/ tests/
ruff format src/ tests/
pyright src/
pytest -q
```

- The repo pre-commit hook prefers a repo-local `.venv` Ruff binary when available and falls back to `PATH`.

## Provider Notes

- Conversation state still preserves `previous_response_id`, `response_id_history`, `prompt_cache_key`, and `grok_conv_id`.
- `collections_search` requires `XAI_COLLECTION_IDS`.
- Raw Responses API behavior, retry/backoff handling, and file upload lifecycle now live primarily in `discord_grok.cogs.grok.client`.
- Chat, image, video, and TTS command bodies are delegated from `discord_grok.cogs.grok.cog` into feature modules.
- Remote MCP is configured per `/grok chat` invocation with raw `mcp` and `mcp_allowed_tools` inputs, then persisted as `mcp_servers` on `ChatCompletionParameters`.
- `resolve_selected_tools()` skips the canonical `mcp` marker and only emits MCP tool payloads from validated `mcp_servers`, preventing duplicate MCP entries.
- MCP is intentionally excluded from the built-in tool dropdown so dropdown changes only affect built-in tools.
- Attachment size limits (from `discord_grok.cogs.grok.attachments`): images are capped at 20 MB (`MAX_IMAGE_SIZE`), other files at 48 MB (`MAX_FILE_SIZE`). Patch these constants when writing upload tests.
