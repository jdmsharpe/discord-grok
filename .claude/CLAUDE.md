# Discord Grok Bot - Developer Reference

## Quick Start

```bash
# Install dependencies (uv-managed: creates .venv from uv.lock; no pip inside — use `uv pip` if needed)
uv sync --extra dev

# Copy and fill in environment variables
cp .env.example .env

# Enable repo pre-commit hook (one-time setup)
git config core.hooksPath .githooks

# Run the bot
uv run python src/bot.py

# Or with Docker
docker compose up --build
```

## Gotchas

- Uses **`py-cord`** (not `discord.py`). The slash-command API differs; don't mix docs between the two.
- `GUILD_IDS` must list at least one guild ID. Empty or unset parses to `[]`, never `None`, and py-cord only treats `guild_ids is None` as global — so the commands register **nowhere**: not globally, not per-guild. Nothing fails at startup; the commands simply never appear.

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `BOT_TOKEN` | Yes | Discord bot token |
| `XAI_API_KEY` | Yes | xAI API key for Grok requests |
| `GUILD_IDS` | Yes | Comma-separated Discord guild IDs for slash command registration; empty or unset registers no commands at all (not enforced by `validate_required_config()`) |
| `XAI_COLLECTION_IDS` | No | Comma-separated collection IDs; enables `collections_search` tool |
| `XAI_MCP_PRESETS_JSON` | No | Inline JSON object of named HTTPS MCP presets for `/grok chat` |
| `XAI_MCP_PRESETS_PATH` | No | Path to a JSON file containing named HTTPS MCP presets |
| `SHOW_COST_EMBEDS` | No | Show token/cost embeds on responses (default: `true`; accepts `true/1/yes`) |
| `XAI_PRICING_PATH` | No | Override the bundled `src/discord_grok/config/pricing.yaml` |
| `LOG_FORMAT` | No | `text` (default) or `json` for structured JSON-lines output |
| `SAFETY_IDENTIFIER_SECRET` | No | HMAC key for the per-user identifier (chat `safety_identifier`, image `user`; default: derived from `BOT_TOKEN`); see the `safety_identifier` provider note |

`validate_required_config()` raises `RuntimeError` at startup for missing/blank `BOT_TOKEN` or `XAI_API_KEY`, and when `SAFETY_IDENTIFIER_SECRET` equals `XAI_API_KEY` (the message leaves out both values).

## Slash Commands

Commands are split across three groups to keep each top-level command under Discord's 8000-byte payload limit:

| Command | Description |
| --- | --- |
| `/grok chat` | Start a conversation with Grok (supports advanced tuning, tools, preset-backed MCP, and file attachments) |
| `/grok check_permissions` | Verify bot permissions in the current channel |
| `/grok-media image` | Generate or edit an image |
| `/grok-media video` | Generate a video from a prompt or image |
| `/grok-tools tts` | Convert text to speech audio |

`/grok chat` currently exposes:

- Core inputs: `prompt`, `system_prompt`, `model`, `attachment`
- Model tuning: `max_tokens`, `temperature` (0–1), `top_p`, `frequency_penalty`, `presence_penalty`, `reasoning_effort`, `agent_count` (multi-agent model only; sent as `reasoning.effort`, see the provider note), `service_tier` (priority processing, see the provider note)
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
    ├── cost_line.py                 # One-line cost formatter, byte-identical across the discord-* bots (change all five together)
    ├── logging_setup.py             # Structured logging + request-id ContextVar
    ├── stickers.py                  # Sticker-to-text labels for follow-ups, byte-identical across the discord-* bots (change all five together)
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
        ├── command_options.py        # Chat model catalog + slash-command option choices
        ├── embed_delivery.py         # Discord embed batching + send helpers (6000-char/10-embed limits); byte-identical across the five AI bots
        ├── embeds.py
        ├── image.py
        ├── models.py
        ├── responses.py
        ├── safety.py                 # HMAC-SHA256 per-user identifier (Responses safety_identifier, image user)
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
- The test suite is organized into module-aligned files: `test_grok_cog`, `test_grok_chat`, `test_grok_client`, `test_grok_commands`, `test_choice_caps`, `test_grok_tooling`, `test_grok_embeds`, `test_grok_responses`, `test_grok_safety`, `test_grok_state`, `test_embed_delivery`, `test_button_view`, `test_config_auth`, `test_config_pricing`, `test_logging_setup`, `test_lazy_imports`, `test_config_mcp`, `test_option_defaults`, `test_package_import`, and `test_util`.
- MCP preset coverage lives in `tests/test_config_mcp.py`.
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
  - `discord_grok.cogs.grok.safety.SAFETY_IDENTIFIER_KEY` (set to `None` to omit `safety_identifier` and the image `user`, or to a test key)
- Import `GrokCog` from `discord_grok`; do not reintroduce legacy `xai_api` shim paths.

## Validation Commands

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pyright src/
uv run pytest -q
```

- The repo pre-commit hook (`.githooks/pre-commit`) runs `ruff format` (auto-applied + re-staged), then `ruff check` (blocking), then `pyright` and `pytest --collect-only` as warning-only smoke tests. Resolves tools from `.venv/bin` or `.venv/Scripts` first, then `PATH`.

## Provider Notes

Probe results, dates, token counts and the history behind each bold-titled bullet are in [provider-history.md](provider-history.md) under the same title, together with the log of past SDK sweeps. Read the matching entry before changing that area; add new evidence there and keep each bullet here to the rule, the constants to keep aligned and the guarding tests.

- Conversation state still preserves `previous_response_id`, `response_id_history`, `prompt_cache_key`, and `grok_conv_id`.
- `collections_search` requires `XAI_COLLECTION_IDS`.
- **Raw HTTP chat and SDK migration:** raw Responses API calls, retry/backoff and the file upload lifecycle live in `discord_grok.cogs.grok.client`. Chat (and TTS) use raw HTTP because xai-sdk has no Responses API surface and no `prompt_cache_key`; image and video use the SDK. Once both exist, move `chat.py` / `client.py` onto the SDK. (Last checked xai-sdk 1.21.0 on 2026-11-06: `AsyncClient` has no `responses` attribute and `chat.create` has no `prompt_cache_key`, so the blocker remains; `chat.create` now accepts `safety_identifier`, which the bot already sends over raw HTTP. Available and not wired: `video.generate`'s `last_frame_url` / `last_frame_file_id` / `keyframes` (grok-imagine-video-1.5 only) and `reference_audios` / `generate_audio`; Imagine file storage (`storage_options` + `public_url`, which would let the bot post a URL instead of downloading and re-uploading bytes); Files-API `file_id` inputs; `compact_context()`. Recheck on the next minor bump.)
- Chat, image, video, and TTS command bodies are delegated from `discord_grok.cogs.grok.cog` into feature modules.
- Remote MCP is configured per `/grok chat` invocation with comma-separated preset names. Presets are loaded from `XAI_MCP_PRESETS_JSON` and `XAI_MCP_PRESETS_PATH`, validated at config-load time, and then persisted as `mcp_servers` on `ChatCompletionParameters`.
- Each MCP preset supports `url` (required HTTPS), `authorization_env_var` (optional), and `allowed_tools` (optional).
- `resolve_selected_tools()` skips the canonical `mcp` marker and only emits MCP tool payloads from validated `mcp_servers`, preventing duplicate MCP entries.
- MCP is intentionally excluded from the built-in tool dropdown so dropdown changes only affect built-in tools.
- The slash-command surface no longer accepts X handle filters or web domain allow/block lists; only media toggles and `x_search_date_range` remain pre-start search refinements.
- Attachment size limits (from `discord_grok.cogs.grok.attachments`): images are capped at 20 MB (`MAX_IMAGE_SIZE`), other files at 48 MB (`MAX_FILE_SIZE`). Patch these constants when writing upload tests.
- **Sampling range:** the `temperature` option advertises and enforces xAI's 0–1 range.
- **Priority processing:** `/grok chat service_tier:priority` requests priority scheduling and persists across follow-ups. The `service_tier` xAI returns decides whether the 2x token and cache estimate applies; a default or unreported tier keeps normal rates, and tool fees are never multiplied. A reported `usage.cost_in_usd_ticks` still takes precedence over the estimate.
- **Image and video cost:** prefer `result.cost_usd` reported by xai-sdk (the server's `cost_in_usd_ticks`, which already includes xAI's per-input-image surcharge on edits and image-to-video); fall back to bundled YAML pricing only when it is absent. Mock `cost_usd=None` in tests to exercise the fallback. TTS has no SDK cost: `calculate_tts_cost` always bills from YAML. In the YAML fallback each `image_generation` / `video_generation` row declares `per_input_image` (loaded into `IMAGE_INPUT_PRICING` / `VIDEO_INPUT_IMAGE_PRICING`; the `unknown_*` fallbacks use $0.01, the highest published rate, so an unknown model is not under-billed), and `calculate_image_cost(..., input_images=N)` / `calculate_video_cost(..., input_images=N)` add it; `image.py` passes 1 when editing and `video.py` when an attachment is supplied. Guards: `test_bundled_yaml_loads_input_image_surcharges` (`tests/test_config_pricing.py`), the `*_adds_input_image_surcharge*` cases in `tests/test_util.py`, the `*_yaml_fallback_adds_input_image_surcharge` command tests, `test_image_cost_prefers_sdk_reported_cost_usd` / `test_video_cost_prefers_sdk_reported_cost_usd` (`tests/test_grok_commands.py`).
- **Chat cost:** `extract_usage` (`responses.py`) reads `usage.cost_in_usd_ticks` (1e-10 USD) into `cost_usd`, and both chat paths in `chat.py` bill that figure when present, falling back to `calculate_cost` + `calculate_tool_cost`. `extract_tool_usage` reads tool counts from `usage.server_side_tool_usage_details` (`x_search_calls`, `web_search_calls`, `code_interpreter_calls`, `file_search_calls`, `mcp_calls`, `document_search_calls`, `image_generation_calls` → `SERVER_SIDE_TOOL_*` keys) and still honours the legacy top-level `server_side_tool_usage` map. The cost line shows tools only as call counts (labels in `TOOL_USAGE_DETAIL_LABELS`, `tooling.py`); their charges are inside the request cost. xAI's `output_tokens` already includes `reasoning_tokens` (docs.x.ai says otherwise), so `calculate_cost` bills `output_tokens` once and `append_pricing_embed` shows `reasoning_tokens` in parentheses. X Search bills per post and per user profile fetched, so the per-call `SERVER_SIDE_TOOL_X_SEARCH` row is only a fallback estimate. Guards: `tests/test_grok_chat.py::test_chat_cost_line_shows_the_reported_cost_added_to_the_daily_total`, `tests/test_grok_responses.py::test_extract_usage_with_details`; `extract_tool_usage` has no direct unit test.
- **grok-4.7:** the `/grok chat` default. $2.00 input / $0.50 cached / $6.00 output per MTok, and $4.00 / $1.00 / $12.00 for every token once a prompt reaches 200k tokens, so the catalog entry uses the `grok_4_6` pricing class. 500k context; text and image input. `reasoning_effort` accepts `low` / `medium` / `high` / `xhigh`; `none` returns 400 and an unset effort runs at `high`. Penalties are gated as on grok-4.6: the catalog entry has no `supports_penalties`, so `chat.py` rejects `frequency_penalty` / `presence_penalty` before the request; enabling them would also need the "Non-reasoning models only" option descriptions and `test_penalty_supported_models_are_non_reasoning` changed. `temperature` is not gated for any model. Responses carry `reasoning.encrypted_content` even when not requested; `extract_response_text` reads only `summary_text` parts. grok-4.6 stays selectable. Guards: `test_default_chat_model_is_grok_4_7` and `test_grok_4_7_entry_matches_its_published_spec` (`tests/test_grok_commands.py`).
- **Reasoning efforts:** grok-4.5, 4.6 and 4.7 accept `low`, `medium`, `high`, `xhigh` (default `high`); grok-4.3 accepts `none` through `xhigh` (default `low`). xAI rejects an unknown effort with 400 "Invalid reasoning effort." rather than mapping it to `high`. `xhigh` is in the `reasoning_efforts` set of all four effort models. Guards: `test_xhigh_is_offered_on_every_effort_model` (`tests/test_util.py`), `test_chat_passes_xhigh_effort` (`tests/test_grok_chat.py`).
- **Multi-agent agent count:** the Responses API has no `agent_count` field, so the `/grok chat` `agent_count` option on `grok-4.20-multi-agent` is sent as `reasoning: {"effort": ...}` via `MULTI_AGENT_EFFORT_BY_AGENT_COUNT` (`client.py`: 4 → `medium`, 16 → `high`). The server ignores a top-level `agent_count`, accepts any effort string on this model without validation and always echoes `medium`, so the echo does not show which setting ran. A 16-agent request costs about 4–6x a 4-agent one (billed from `cost_in_usd_ticks`). `reasoning_effort` stays rejected on the multi-agent model (its catalog entry has no `reasoning_efforts`). Guards: `test_chat_sends_agent_count_as_reasoning_effort_for_multi_agent` and `test_follow_up_sends_agent_count_as_reasoning_effort` (`tests/test_grok_chat.py`).
- **grok-imagine-video-1.5-lite:** selectable, not the default. $0.02 / $0.03 / $0.14 per second at 480p / 720p / 1080p plus $0.01 per input image; text and image input only (no reference audio); 1–15 s; its 1080p output is 720p upscaled. Billed from the SDK's `cost_usd` like the other video models. Guards: `test_video_1_5_lite_is_offered_and_not_the_default` and `test_video_1_5_lite_sends_its_id_and_bills_its_own_rate` (`tests/test_grok_commands.py`), and the Lite rows in `test_bundled_yaml_loads_video_pricing` / `test_calculate_video_cost_per_resolution`.
- **`safety_identifier`:** both chat paths in `chat.py` pass `build_safety_identifier(<author id>)` (`safety.py`) to `build_responses_payload`, which sets the top-level `safety_identifier` of the POST /v1/responses body; `image.py` sends the same value as `user=` on `image.sample` / `image.sample_batch` (generation and editing). The value is `hmac.new(key, str(discord_user_id), sha256).hexdigest()` (64 hex characters, not truncated). Key: `SAFETY_IDENTIFIER_SECRET` (stripped) when set, otherwise `HMAC-SHA256(BOT_TOKEN, b"safety-identifier-v1")`, computed once at import into `SAFETY_IDENTIFIER_KEY`; with neither, no identifier is sent (never an unkeyed hash, since Discord user IDs are enumerable). The construction, env var and label are shared with discord-claude, discord-openai and discord-openrouter; change all four together. Never key it with `XAI_API_KEY`; `validate_required_config()` rejects a secret equal to it. Rotating `BOT_TOKEN` without a secret changes every user's value. Not sent on POST /v1/tts (documents neither field), `video.generate` (no such parameter) or `files.upload` / `files.delete`. The SDK's image telemetry copies `user` into a `user_id` span attribute unless `XAI_SDK_DISABLE_SENSITIVE_TELEMETRY_ATTRIBUTES` is set; the bot installs no tracer provider, so those spans are not recorded. Nothing logs the identifier or the key. Guards: `tests/test_grok_safety.py` (including `test_fixed_vectors_shared_with_sibling_bots` and `test_module_key_is_read_from_the_environment`), and `test_safety_identifier_secret_is_optional_and_stripped` / `test_validate_required_config_rejects_the_api_key_as_safety_secret` (`tests/test_config_auth.py`).
- `fallbacks.unknown_image_model.per_image` is $0.08 (Image 2.0 at 2k/medium, the dearest published tier, 2026-09-15); it was $0.07 before v1.10.0.
- **Resolution-based media pricing:** Grok Imagine bills per output resolution, so `image_generation` / `video_generation` rows in `pricing.yaml` map resolution → rate and `calculate_image_cost` / `calculate_video_cost` take the requested resolution. `grok-imagine-image-2.0` also bills by quality: its rows are keyed `"<resolution>/<quality>"` and `calculate_image_cost` bills the most expensive tier for a resolution when no quality is supplied. 1080p is priced only for `grok-imagine-video-1.5-preview` and `grok-imagine-video-1.5-lite`, so `_validate_video_resolution` (`video.py`) rejects it on other models. A scalar rate bills flat. When no resolution is passed, the tier assumed is `DEFAULT_IMAGE_RESOLUTION` / `DEFAULT_VIDEO_RESOLUTION` in `tooling.py`, which must match the `/grok-media` option defaults.
- **Image Quality retirement:** the Quality and former Pro IDs were removed from menu/allowlist on 2026-10-28 ahead of the November 2 retirement; retain their historical pricing rows. Image 2.0 remains the default.
- **Stickers are sent as text:** `handle_new_message_in_conversation` (`chat.py`) sends each sticker in `message.stickers` as a `[Sticker: <name>]` line after the message text, built by `text_with_stickers` in `stickers.py` (byte-identical in the five AI bots; change all five together). The sticker image is not sent. A follow-up with nothing to send is ignored and the conversation stays open. Guards: `tests/test_stickers.py` and the sticker follow-up tests in `tests/test_grok_chat.py`.

## Runtime Conventions (Cross-Project)

- **Pricing:** loaded from `src/discord_grok/config/pricing.yaml` by `config/pricing.py` at import time. Chat pricing uses a class indirection (`MODEL_PRICING_CLASSES`) joined to `CHAT_MODEL_CATALOG` in `command_options.py`. Override the YAML via `XAI_PRICING_PATH`. [docs.x.ai](https://docs.x.ai/docs/models) is canonical; [genai-prices/x_ai.yml](https://github.com/pydantic/genai-prices/blob/main/prices/providers/x_ai.yml) is a secondary cross-reference that is known to lag for xAI, so verify prices against docs.x.ai first. In the model JSON docs.x.ai embeds, only the us-east-1 rows carry the base price; us-central-1 rows include a 1.1x regional premium. Server-side tool rows are catalog-fallback estimates; the reported `cost_in_usd_ticks` takes precedence when present.
- **Retry**: `client.post_with_retries` wraps every xAI HTTP call with exponential backoff + jitter. `MAX_API_ATTEMPTS=5`, honors `Retry-After` on 429, retries on `RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}` and transport errors.
- **Conversation TTL**: `prune_runtime_state` in `cogs/grok/state.py` evicts conversations older than `CONVERSATION_TTL` (12h) every 15 minutes via `@tasks.loop`. Caps at `MAX_ACTIVE_CONVERSATIONS`. Daily costs retained for `DAILY_COST_RETENTION_DAYS` (30).
- **Request IDs**: `cog_before_invoke` (and `on_message`) bind a fresh 8-char hex id via `discord_grok.logging_setup.bind_request_id`. All downstream `logger.info`/`warning`/`error` calls automatically include the id. Set `LOG_FORMAT=json` for JSON-lines output.
- **Async file I/O**: blocking `open()` and `pathlib` methods (`read_bytes`, `write_bytes`, `unlink`, etc.) inside `async def` freeze the Discord event loop and stall every concurrent slash command. Wrap them with `asyncio.to_thread(...)` so the I/O runs on a worker thread. Enforced by `ruff` (`ASYNC230`/`ASYNC240`).
