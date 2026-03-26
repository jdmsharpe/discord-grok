# Discord Grok Bot - Claude Code Context

## Repository Overview

Discord bot (Pycord 2.x) integrating xAI APIs: multi-turn chat (Responses API via aiohttp), image/video generation (xAI SDK/gRPC), and TTS (REST API).

Chat uses `previous_response_id` for server-side conversation state with automatic cached-token billing. Image, video, and file upload/delete use the `xai_sdk` Python package.

Four optional tools in `/grok chat`:

- `web_search`
- `x_search`
- `code_execution` (mapped to `code_interpreter` in the Responses API)
- `collections_search` (mapped to `file_search` in the Responses API; requires `XAI_COLLECTION_IDS`)

## Project Structure

```text
discord-grok/
├── .github/workflows/main.yml   # CI workflow (pytest)
├── src/
│   ├── bot.py                   # Main bot entry point
│   ├── xai_api.py               # xAI Discord cog (`/grok` commands)
│   ├── button_view.py           # Conversation buttons + tool select dropdown
│   ├── util.py                  # Dataclasses, tool constants, helpers
│   └── config/auth.py           # Env var loading
├── tests/
│   ├── conftest.py
│   ├── test_xai_api.py
│   ├── test_button_view.py
│   └── test_util.py
├── Dockerfile, Dockerfile.test, docker-compose.yaml
├── requirements.txt
└── README.md
```

## Key Architecture Details

These are non-obvious patterns worth knowing when modifying the codebase:

- **Tool type mapping**: Our canonical names differ from the Responses API names. `code_execution` maps to `{"type": "code_interpreter"}`, `collections_search` maps to `{"type": "file_search", "vector_store_ids": [...]}`. The reverse mapping lives in `_TOOL_TYPE_TO_CANONICAL` in `util.py`.
- **`TOOL_BUILDERS`** only covers 3 tools (`web_search`, `x_search`, `code_execution`). `collections_search` is built inline in `resolve_selected_tools()` because it needs `collection_ids`.
- **`resolve_selected_tools()`** is a standalone function in `util.py` (not a cog method). The cog has a thin wrapper that passes `XAI_COLLECTION_IDS`. This breaks the circular coupling with `button_view.py`.
- **Agentic state**: `include: ["reasoning.encrypted_content"]` + `store: true` is set for all tool-using conversations AND multi-agent models. This enables server-side state via `previous_response_id`.
- **`prompt_cache_key`**: `Conversation.prompt_cache_key` defaults to `""`. Callers always pass a `uuid.uuid4()` string explicitly — there is no default factory.
- **Multi-agent constraints**: `agent_count` is only valid for `MULTI_AGENT_MODELS`. Multi-agent models reject `max_tokens`.
- **File attachments**: Images go inline (`input_image`); non-image files are uploaded via `client.files.upload()` and sent as `input_file`. Files are cleaned up on conversation end AND on chat error (orphaned file cleanup in the exception handler).
- **`SHOW_COST_EMBEDS`** is checked at each call site, not inside the embed helper functions.
- **`TOOL_USAGE_DISPLAY_NAMES`** includes server-side tool types beyond the four user-selectable tools (e.g., `VIEW_X_VIDEO`, `VIEW_IMAGE`, `MCP`, `ATTACHMENT_SEARCH`).
- **Session lifecycle**: `aiohttp.ClientSession` is lazily created via `_get_http_session()` and cleaned up via both `cog_unload()` (cog removal) and `on_close()` (bot shutdown).
- **Logging**: `logging.basicConfig()` is called once in `bot.py`, not in the cog constructor.
- **Env vars**: `BOT_TOKEN` and `XAI_API_KEY` use `os.environ[]` (fail-fast `KeyError` on missing). Optional vars use `os.getenv()` with defaults.
- **Type checking**: The codebase passes `pyright` with 0 errors. `button_view.py` uses `from __future__ import annotations` + `TYPE_CHECKING` for the `xAIAPI` forward reference.

## `/grok chat` Parameters

Current parameter count: 23

1. `prompt`
2. `system_prompt`
3. `model`
4. `attachment` (images passed inline, other files uploaded via xAI Files API)
5. `max_tokens` (not supported by multi-agent models)
6. `temperature`
7. `top_p`
8. `frequency_penalty`
9. `presence_penalty`
10. `reasoning_effort` (choices: low, high; only `grok-3-mini`; default: not set)
11. `agent_count` (choices: 4, 16; only multi-agent models; default: not set)
12. `web_search`
13. `x_search`
14. `code_execution`
15. `collections_search`
16. `x_search_images` (enable image understanding in X posts)
17. `x_search_videos` (enable video understanding in X posts)
18. `x_search_date_range` (comma-separated ISO8601 start,end date filter)
19. `x_search_allowed_handles` (comma-separated, max 10, mutually exclusive with excluded)
20. `x_search_excluded_handles` (comma-separated, max 10, mutually exclusive with allowed)
21. `web_search_allowed_domains` (comma-separated, max 5, mutually exclusive with excluded)
22. `web_search_excluded_domains` (comma-separated, max 5, mutually exclusive with allowed)
23. `web_search_images` (enable image understanding during web browsing)

## `/grok image` Parameters

Current parameter count: 6

1. `prompt`
2. `model` (choices: grok-imagine-image-pro (default), grok-imagine-image)
3. `aspect_ratio` (13 choices matching `ImageAspectRatio`; default: 1:1)
4. `resolution` (choices: 1k, 2k; default: not set (API default 1k))
5. `count` (1-10 images via `sample_batch`; default: 1; not supported in Image Editing mode)
6. `attachment` (image to edit/remix; triggers Image Editing mode; passes Discord CDN URL as `image_url` to SDK)

## `/grok video` Parameters

Current parameter count: 5

1. `prompt`
2. `aspect_ratio` (7 choices matching `VideoAspectRatio`; default: 16:9)
3. `duration` (1-15 seconds; default: 5)
4. `resolution` (choices: 720p (default), 480p)
5. `attachment` (image for first frame; triggers Image-to-Video mode; passes Discord CDN URL as `image_url` to SDK)

## `/grok tts` Parameters

Current parameter count: 6

1. `text` (max 15,000 characters; supports speech tags like `[pause]`, `<whisper>`)
2. `voice` (choices: eve, ara, rex, sal, leo; default: eve)
3. `language` (free-text BCP-47 code or `auto`; default: auto)
4. `output_format` (choices: mp3, wav, pcm, mulaw, alaw; default: mp3)
5. `sample_rate` (choices: 8000, 16000, 22050, 24000, 44100, 48000 Hz; default: 24000)
6. `bit_rate` (choices: 32000, 64000, 96000, 128000, 192000 bps; MP3 only; default: 128000)

## Environment Variables

| Variable             | Description                                                                |
| -------------------- | -------------------------------------------------------------------------- |
| `BOT_TOKEN`          | Discord bot token                                                          |
| `GUILD_IDS`          | Comma-separated Discord guild IDs                                          |
| `XAI_API_KEY`        | xAI API key for chat/image/video/tts                                       |
| `XAI_COLLECTION_IDS` | Optional comma-separated collection IDs used by `collections_search`       |
| `SHOW_COST_EMBEDS`   | Show cost/token usage embeds on responses (`true`/`false`, default `true`) |

If `collections_search=true` and `XAI_COLLECTION_IDS` is empty, chat returns a user-facing error.

## Test Commands

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Notes for Future Changes

- Tool support targets exactly four built-in tools for `/grok chat`.
- If new tools are added:
  - Update `src/util.py` constants/builders and `resolve_selected_tools()`
  - Update tool select options in `src/button_view.py` (`max_values` is derived automatically from `len(AVAILABLE_TOOLS)`)
  - Update README + this file
  - Add/adjust tests in `tests/test_xai_api.py`, `tests/test_button_view.py`, and `tests/test_util.py`
- Run `pyright src/` to verify type safety before committing.
