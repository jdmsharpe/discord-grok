# Discord Grok Bot - Claude Code Context

## Repository Overview

This is a Discord bot built on Pycord 2.x that integrates xAI's SDK to provide:

- Multi-turn conversational chat with Grok
- Built-in tool calling in `/grok chat`
- Image generation
- Video generation
- Text-to-speech audio generation

The conversation path supports optional tool calling with:

- `web_search`
- `x_search`
- `code_execution`
- `collections_search` (requires `XAI_COLLECTION_IDS`)

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
  - `MODEL_PRICING` maps chat models to `(input_cost, output_cost)` per million tokens
  - `IMAGE_PRICING` maps image models to flat per-image cost
  - `VIDEO_PRICING_PER_SECOND` flat per-second video cost
  - `calculate_cost()`, `calculate_image_cost()`, `calculate_video_cost()`
- `ChatCompletionParameters`
  - Stores conversational model settings and Discord conversation metadata
  - Includes `tools` for active tool configuration in ongoing conversations
- `Conversation`
  - Stores the pair: `params` + mutable `chat` object from `xai_sdk`
  - Tracks `file_ids` for xAI Files API cleanup on conversation end
- Tool helpers
  - `TOOL_WEB_SEARCH`, `TOOL_X_SEARCH`, `TOOL_CODE_EXECUTION`, `TOOL_COLLECTIONS_SEARCH`
  - `AVAILABLE_TOOLS` maps tool constants to display names for the Discord UI
  - `TOOL_BUILDERS` for tool proto creation
  - `resolve_tool_name()` to map tool protos back to canonical names
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
- Tool flow:
  - `resolve_selected_tools()` builds tool protos
  - `collections_search` is guarded by `XAI_COLLECTION_IDS`
  - `chat.create(... include=["inline_citations"])` enabled in chat
  - `CitationInfo` TypedDict stores `url` + `source` (`"web"`, `"x"`, `"collections"`)
  - `extract_tool_info()` prefers `response.inline_citations` (structured web/x citation objects), falls back to `response.citations` with URL-based classification
  - `append_sources_embed()` groups citations by source type with headings when mixed (Web, X Posts, Collections)
  - `_apply_tools_to_chat()` updates tools dynamically when toggled mid-conversation
- Multi-agent flow:
  - Multi-agent models automatically get `use_encrypted_content=True` for proper multi-turn context
  - `agent_count` (4 or 16) is passed to `chat.create()` when specified
  - `max_tokens` is rejected for multi-agent models (not supported by the API)
  - `agent_count` is rejected for non-multi-agent models
- File attachment flow (xAI Files API):
  - Non-image attachments are downloaded from Discord and uploaded via `client.files.upload()`
  - `_upload_file_attachment()` handles download + upload, enforces 48 MB Files API limit
  - Inline images are validated: only `image/jpeg` and `image/png` accepted, max 20 MiB
  - Images are sent with `detail="high"` for better image understanding
  - Uploaded file IDs are tracked in `Conversation.file_ids`
  - `xai_file(file_id)` is included in message content parts (triggers server-side `attachment_search`)
  - `_cleanup_conversation_files()` deletes all tracked files from xAI on conversation end
  - `end_conversation()` removes the conversation and cleans up files
- TTS flow (xAI TTS REST API):
  - `_generate_tts()` calls `POST https://api.x.ai/v1/tts` directly via aiohttp (not part of xAI SDK)
  - Returns raw audio bytes; sent to Discord as a file attachment
  - Supports 5 voices: `eve`, `ara`, `rex`, `sal`, `leo`
  - Supports BCP-47 language codes (default `en`)
  - Output formats: `mp3` (default), `wav`
  - Text limit: 15,000 characters
  - Constants `TTS_API_URL` and `TTS_MAX_CHARS` defined in `xai_api.py`
- Pricing and token usage:
  - Token usage extracted via `response.usage.prompt_tokens` / `response.usage.completion_tokens` (xAI SDK uses OpenAI-style naming, not `input_tokens`/`output_tokens`)
  - `append_pricing_embed()` shows per-request cost, token counts, and daily cumulative cost for chat
  - `append_generation_pricing_embed()` shows flat cost for image/video generation
  - `SHOW_COST_EMBEDS` is checked at each call site (not inside the helper functions)
  - Sources and cost embeds are included in the main response message (after response/reasoning embeds, before the ButtonView)
  - `_track_daily_cost()` accumulates token-based costs per `(user_id, date)`
  - `_track_daily_cost_flat()` accumulates flat costs (image/video) per `(user_id, date)`
  - `self.daily_costs` dict keyed by `(user_id, date_iso_str)`

### `src/button_view.py`

UI controls attached to conversation messages:

- Buttons:
  - Regenerate
  - Pause/Resume
  - Stop
- Select dropdown:
  - Toggle tools per conversation (`web_search`, `x_search`, `code_execution`, `collections_search`)
  - Calls cog-level `resolve_selected_tools()`
  - Applies tool changes immediately to the active chat object

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

Current parameter count: 4

1. `text` (max 15,000 characters)
2. `voice` (choices: eve, ara, rex, sal, leo; default: eve)
3. `language` (free-text BCP-47 code; default: en)
4. `output_format` (choices: mp3, wav; default: mp3)

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
