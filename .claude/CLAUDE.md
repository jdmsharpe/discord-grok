# Discord Grok Bot - Developer Reference

## Supported Entry Points

- Launcher: `python src/bot.py` remains supported and delegates to `discord_grok.bot.main`.
- Cog composition contract:

  ```python
  from discord_grok import xAIAPI

  bot.add_cog(xAIAPI(bot=bot))
  ```

- Legacy shim: `src/xai_api.py` exists only for import compatibility and emits a `DeprecationWarning`.

## Package Layout

```text
src/
├── bot.py                           # Thin repo-local launcher
├── xai_api.py                       # Temporary compatibility shim
├── config/                          # Repo-local compatibility shim
└── discord_grok/
    ├── __init__.py
    ├── bot.py
    ├── config/
    │   ├── __init__.py
    │   └── auth.py
    └── cogs/grok/
        ├── __init__.py
        ├── client.py
        ├── cog.py
        ├── embeds.py
        ├── models.py
        ├── tooling.py
        └── views.py
tests/
├── conftest.py
├── fixtures.py
└── ...
```

Top-level `button_view.py`, `util.py`, and `config/` remain repo-local compatibility layers and are not the installed public API.

## Testing And Patch Targets

- `pytest` runs with `pythonpath = ["src"]`.
- Shared response payloads now live in `tests/fixtures.py`; do not rely on bare `conftest` imports for data fixtures.
- New tests and patches should target real owners under `discord_grok...`, not `xai_api`.
- Examples:
  - `discord_grok.cogs.grok.cog.RESPONSES_API_URL`
  - `discord_grok.cogs.grok.client.RETRYABLE_STATUS_CODES`
  - `discord_grok.cogs.grok.tooling.resolve_tool_name`
  - `discord_grok.cogs.grok.views.ButtonView`

## Validation Commands

```bash
ruff check src/ tests/
ruff format src/ tests/
pyright src/
pytest -q
```

## Provider Notes

- Conversation state still preserves `previous_response_id`, `response_id_history`, `prompt_cache_key`, and `grok_conv_id`.
- `collections_search` requires `XAI_COLLECTION_IDS`.
- Raw Responses API behavior, retry/backoff handling, and file upload lifecycle live under `discord_grok.cogs.grok`.
