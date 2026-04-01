# Discord Grok Bot - Developer Reference

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
- The test suite is organized into module-aligned files such as `tests/test_grok_cog.py`, `tests/test_grok_chat.py`, `tests/test_grok_client.py`, `tests/test_grok_commands.py`, `tests/test_grok_tooling.py`, `tests/test_config_auth.py`, and `tests/test_lazy_imports.py`.
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
