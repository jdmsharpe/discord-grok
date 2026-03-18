# Discord Grok Bot - Claude Code Context

## Repository Overview

This is a Discord bot built on Pycord 2.x that integrates xAI's APIs to provide:

- Multi-turn conversational chat with Grok (via xAI Responses API)
- Built-in tool calling in `/grok chat`
- Image generation (via xAI SDK)
- Video generation (via xAI SDK)
- Text-to-speech audio generation (via xAI TTS REST API)

The chat path calls the xAI Responses API (`POST /v1/responses`) directly via
aiohttp.  Multi-turn conversations use `previous_response_id` for server-side
state management with automatic billing optimization (cached tokens).  Image,
video, and file upload/delete still use the `xai_sdk` Python package (gRPC).

The conversation path supports optional tool calling with:

- `web_search`
- `x_search`
- `code_execution` (mapped to `code_interpreter` in the Responses API)
- `collections_search` (mapped to `file_search` in the Responses API; requires `XAI_COLLECTION_IDS`)

## Project Structure

```text
discord-grok/
├── .github/
│   └── workflows/
│       └── main.yml          # CI workflow (pytest)
├── src/
│   ├── bot.py               # Main bot entry point
│   ├── xai_api.py           # xAI Discord cog and slash commands
│   ├── button_view.py       # Conversation buttons + tool select dropdown
│   ├── util.py              # Dataclasses, tool constants, helper utilities
│   └── config/
│       └── auth.py          # BOT_TOKEN, GUILD_IDS, XAI_API_KEY, XAI_COLLECTION_IDS
├── tests/
│   ├── conftest.py
│   ├── test_xai_api.py
│   ├── test_button_view.py
│   └── test_util.py
├── .env.example
├── .gitattributes
├── requirements.txt
├── Dockerfile
├── Dockerfile.test
├── docker-compose.yaml
└── README.md
```

## Key Components

### `src/util.py`

- Model lists
  - `GROK_MODELS` list of all chat model IDs
  - `GROK_IMAGE_MODELS` list of image generation model IDs
  - `GROK_VIDEO_MODELS` list of video generation model IDs
  - `TTS_VOICES` list of available TTS voice IDs (`eve`, `ara`, `rex`, `sal`, `leo`)
  - `PENALTY_SUPPORTED_MODELS` set of models that accept `frequency_penalty`/`presence_penalty` (non-reasoning only)
  - `REASONING_EFFORT_MODELS` set of models that accept `reasoning_effort` (`grok-3-mini` only)
  - `MULTI_AGENT_MODELS` set of models that support `agent_count` and have special constraints (no `max_tokens`)
- Pricing
  - `MODEL_PRICING` maps chat models to `(input_cost, cached_input_cost, output_cost)` per million tokens
  - `IMAGE_PRICING` maps image models to flat per-image cost
  - `VIDEO_PRICING_PER_SECOND` flat per-second video cost
  - `TOOL_INVOCATION_PRICING` maps `SERVER_SIDE_TOOL_*` keys to per-1k-invocations cost ($5 web/x/code, $10 attachment, $2.50 collections)
  - `TTS_PRICING_PER_MILLION_CHARS` TTS cost at $4.20 per million characters
  - `calculate_cost()` includes `reasoning_tokens` (billed at output rate) and `cached_tokens` (billed at discounted cached rate) parameters
  - `calculate_tool_cost()` sums per-invocation costs from `server_side_tool_usage` dict
  - `calculate_tts_cost()` character-based TTS cost
  - `calculate_image_cost()`, `calculate_video_cost()`
- `ChatCompletionParameters`
  - Stores conversational model settings and Discord conversation metadata
  - Includes `tools` (list of JSON dicts) for active tool configuration in ongoing conversations
- `Conversation`
  - Stores `params`, `previous_response_id`, `response_id_history`, and `file_ids`
  - `previous_response_id` is passed to the Responses API for multi-turn context
  - `response_id_history` is an ordered list of all response IDs, used for regeneration (rewind)
  - Tracks `file_ids` for xAI Files API cleanup on conversation end
- Tool helpers
  - `TOOL_WEB_SEARCH`, `TOOL_X_SEARCH`, `TOOL_CODE_EXECUTION`, `TOOL_COLLECTIONS_SEARCH`
  - `AVAILABLE_TOOLS` maps tool constants to display names for the Discord UI
  - `TOOL_BUILDERS` returns JSON tool dicts for the Responses API (e.g. `{"type": "web_search"}`)
  - `_TOOL_TYPE_TO_CANONICAL` maps Responses API type names back to canonical names (`code_interpreter` → `code_execution`, `file_search` → `collections_search`)
  - `TOOL_USAGE_DISPLAY_NAMES` maps `server_side_tool_usage` keys to human-readable names
  - `resolve_tool_name()` maps JSON tool dicts back to canonical names via `type` key lookup
- Text helpers
  - `chunk_text()`
  - `truncate_text()`
  - `format_xai_error()`

### `src/xai_api.py`

Main Discord cog class: `xAIAPI`

- Command group: `/grok`
- Commands:
  - `/grok chat`
  - `/grok image`
  - `/grok video`
  - `/grok tts`
  - `/grok check_permissions`
- Conversation management:
  - Tracks per-conversation state in `self.conversations`
  - Follow-up messages in the same channel are routed back to the conversation starter
  - Pause/resume and stop controls via `ButtonView`
  - `_strip_previous_view()` removes buttons from the previous turn's message before sending a new response
  - `self.last_view_messages` tracks the most recent message with buttons per user
- Chat API flow (Responses API via aiohttp):
  - `_call_responses_api()` sends `POST /v1/responses` with JSON payload
  - `_build_responses_payload()` constructs the request body (model, input, tools, params)
  - `_build_user_message()` builds JSON user messages (plain text or multimodal with `input_text`/`input_image`/`input_file` parts)
  - `_extract_response_text()` parses `output[]` for assistant text and reasoning summaries; strips `[[N]](url)` citation markers from response text
  - `_extract_usage()` extracts token counts from nested `usage.input_tokens_details` / `usage.output_tokens_details` (with fallback to Chat Completions field names)
  - `RESPONSES_API_URL = "https://api.x.ai/v1/responses"`
- Tool flow:
  - `resolve_selected_tools()` builds JSON tool dicts for the Responses API
  - Tool type mapping: `code_execution` → `{"type": "code_interpreter"}`, `collections_search` → `{"type": "file_search", "vector_store_ids": [...]}`
  - `collections_search` is guarded by `XAI_COLLECTION_IDS`
  - `CitationInfo` TypedDict stores `url` + `source` (`"web"`, `"x"`, `"collections"`)
  - `extract_tool_info()` parses `annotations` array from Responses API output, classifies citations by URL pattern via `_classify_citation_url()`
  - `append_sources_embed()` groups citations by source type with headings when mixed (Web, X Posts, Collections)
  - Mid-conversation tool toggling: `conversation.params.tools` is updated directly; tools are sent fresh with each API call (no proto manipulation needed)
- Agentic state preservation:
  - `include: ["reasoning.encrypted_content"]` + `store: true` is set for all tool-using conversations AND multi-agent models
  - Server-side conversation state via `previous_response_id` — no manual message history management
  - Responses stored for 30 days on xAI servers with automatic billing optimization (cached tokens)
- Multi-agent flow:
  - `agent_count` (4 or 16) is passed in the Responses API payload when specified
  - `max_tokens` is rejected for multi-agent models (not supported by the API)
  - `agent_count` is rejected for non-multi-agent models
- File attachment flow (xAI Files API):
  - Non-image attachments are downloaded from Discord and uploaded via `client.files.upload()`
  - `_upload_file_attachment()` handles download + upload, enforces 48 MB Files API limit
  - Inline images are validated: only `image/jpeg` and `image/png` accepted, max 20 MiB
  - Images are sent as `{"type": "input_image", "image_url": url, "detail": "high"}` content parts
  - Uploaded file IDs are tracked in `Conversation.file_ids`
  - Files are sent as `{"type": "input_file", "file_id": id}` content parts (triggers server-side `attachment_search`)
  - `_cleanup_conversation_files()` deletes all tracked files from xAI on conversation end
  - `end_conversation()` removes the conversation, strips buttons from the last message, cleans up views, and deletes uploaded files
- TTS flow (xAI TTS REST API):
  - `_generate_tts()` calls `POST https://api.x.ai/v1/tts` directly via aiohttp (not part of xAI SDK)
  - Returns raw audio bytes; sent to Discord as a file attachment
  - Supports 5 voices: `eve`, `ara`, `rex`, `sal`, `leo`
  - Supports BCP-47 language codes and `auto` detection (default `auto`)
  - Codecs: `mp3` (default), `wav`, `pcm`, `mulaw`, `alaw`
  - Configurable `sample_rate` (8000–48000 Hz, default 24000) and `bit_rate` (MP3 only, 32000–192000 bps, default 128000)
  - Supports speech tags: inline (`[pause]`, `[laugh]`, etc.) and wrapping (`<whisper>`, `<slow>`, etc.)
  - Text limit: 15,000 characters
  - Constants `TTS_API_URL` and `TTS_MAX_CHARS` defined in `xai_api.py`
  - Cost tracked via `calculate_tts_cost()` and `_track_daily_cost()`; shown via `append_generation_pricing_embed()`
- Pricing and token usage:
  - Token usage extracted via `_extract_usage()` from JSON: `usage.input_tokens`, `usage.output_tokens`, `usage.input_tokens_details.cached_tokens`, `usage.output_tokens_details.reasoning_tokens`, etc.
  - `server_side_tool_usage` extracted from top-level response JSON (e.g. `{"SERVER_SIDE_TOOL_WEB_SEARCH": 3}`)
  - `append_pricing_embed()` shows per-request cost (tokens + tool invocations), token counts (with cached/image/reasoning breakdowns), daily cumulative cost, and optional tool usage line with tool invocation cost
  - `append_generation_pricing_embed()` shows flat cost for image/video/TTS generation
  - `SHOW_COST_EMBEDS` is checked at each call site (not inside the helper functions)
  - Sources and cost embeds are included in the main response message (after response/reasoning embeds, before the ButtonView)
  - `_track_daily_cost()` accumulates any cost (token-based or flat) per `(user_id, date)`; cost is pre-computed at each call site
  - `self.daily_costs` dict keyed by `(user_id, date_iso_str)`
  - Persistent `self.logger.info()` at every API call site (chat x2, image, video, TTS) logging user, model, token counts, cost, and daily total

### `src/button_view.py`

UI controls attached to conversation messages:

- Buttons:
  - Regenerate
  - Pause/Resume
  - Stop
- Select dropdown:
  - Toggle tools per conversation (`web_search`, `x_search`, `code_execution`, `collections_search`)
  - Calls cog-level `resolve_selected_tools()`
  - Updates `conversation.params.tools` directly; next API call uses updated tools

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
18. `x_search_date_range` (comma-separated ISO8601 start,end date filter e.g. YYYY-MM-DD,YYYY-MM-DD)
19. `x_search_allowed_handles` (comma-separated, max 10, mutually exclusive with excluded)
20. `x_search_excluded_handles` (comma-separated, max 10, mutually exclusive with allowed)
21. `web_search_allowed_domains` (comma-separated, max 5, mutually exclusive with excluded)
22. `web_search_excluded_domains` (comma-separated, max 5, mutually exclusive with allowed)
23. `web_search_images` (enable image understanding during web browsing)

## `/grok tts` Parameters

Current parameter count: 6

1. `text` (max 15,000 characters; supports speech tags like `[pause]`, `<whisper>`)
2. `voice` (choices: eve, ara, rex, sal, leo; default: eve)
3. `language` (free-text BCP-47 code or `auto`; default: auto)
4. `output_format` (choices: mp3, wav, pcm, mulaw, alaw; default: mp3)
5. `sample_rate` (choices: 8000, 16000, 22050, 24000, 44100, 48000 Hz; default: 24000)
6. `bit_rate` (choices: 32000, 64000, 96000, 128000, 192000 bps; MP3 only; default: 128000)

## Embed and Truncation Behavior

- `append_response_embeds()`:
  - Truncates very long outputs at ~20,000 chars
  - Chunks into 3,500-char embed segments
- `append_reasoning_embeds()`:
  - Truncates reasoning at ~3,500 chars and wraps in spoiler tags
- Prompt truncation in command metadata embeds:
  - Chat prompt: 2,000 chars
  - Chat system prompt: 500 chars
  - Image prompt: 2,000 chars
  - Video prompt: 2,000 chars
  - TTS text: 2,000 chars
- Sources embed:
  - Up to 8 citation lines per source type
  - Grouped by source type (Web, X Posts, Collections) with headings when multiple types present
  - HTTP citations are rendered as links
  - Non-HTTP citations (e.g., `collections://...`) are rendered as code text

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

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Notes for Future Changes

- Tool support currently targets exactly four built-in tools for `/grok chat`.
- If new tools are added:
  - Update `src/util.py` constants/builders
  - Update `resolve_selected_tools()` in `src/xai_api.py`
  - Update tool select options in `src/button_view.py`
  - Update README + this file
  - Add/adjust tests in `tests/test_xai_api.py`, `tests/test_button_view.py`, and `tests/test_util.py`
