# Discord Grok Bot - Claude Code Context

## Repository Overview

This is a Discord bot built on Pycord 2.x that integrates xAI's SDK to provide:

- Multi-turn conversational chat with Grok
- Built-in tool calling in `/grok converse`
- Image generation
- Video generation

The conversation path supports optional tool calling with:

- `web_search`
- `x_search`
- `code_execution`
- `collections_search` (requires `XAI_COLLECTION_IDS`)

## Project Structure

```text
discord-grok/
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
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
└── README.md
```

## Key Components

### `src/util.py`

- `ChatCompletionParameters`
  - Stores conversational model settings and Discord conversation metadata
  - Includes `tools` for active tool configuration in ongoing conversations
- `Conversation`
  - Stores the pair: `params` + mutable `chat` object from `xai_sdk`
- Tool helpers
  - `TOOL_WEB_SEARCH`, `TOOL_X_SEARCH`, `TOOL_CODE_EXECUTION`, `TOOL_COLLECTIONS_SEARCH`
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
  - `/grok converse`
  - `/grok image`
  - `/grok video`
  - `/grok check_permissions`
- Conversation management:
  - Tracks per-conversation state in `self.conversations`
  - Follow-up messages in the same channel are routed back to the conversation starter
  - Pause/resume and stop controls via `ButtonView`
- Tool flow:
  - `resolve_selected_tools()` builds tool protos
  - `collections_search` is guarded by `XAI_COLLECTION_IDS`
  - `chat.create(... include=["inline_citations"])` enabled in converse
  - `extract_tool_info()` reads `response.citations`
  - `append_sources_embed()` renders source URLs (including `collections://...` citations)
  - `_apply_tools_to_chat()` updates tools dynamically when toggled mid-conversation

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

## `/grok converse` Parameters

Current parameter count: 13

1. `prompt`
2. `system_prompt`
3. `model`
4. `attachment`
5. `max_tokens`
6. `temperature`
7. `top_p`
8. `frequency_penalty`
9. `presence_penalty`
10. `web_search`
11. `x_search`
12. `code_execution`
13. `collections_search`

## Embed and Truncation Behavior

- `append_response_embeds()`:
  - Truncates very long outputs at ~20,000 chars
  - Chunks into 3,500-char embed segments
- `append_reasoning_embeds()`:
  - Truncates reasoning at ~3,500 chars and wraps in spoiler tags
- Prompt truncation in command metadata embeds:
  - Converse prompt: 2,000 chars
  - Converse system prompt: 500 chars
  - Image prompt: 2,000 chars
  - Video prompt: 2,000 chars
- Sources embed:
  - Up to 8 citation lines
  - HTTP citations are rendered as links
  - Non-HTTP citations (e.g., `collections://...`) are rendered as code text

## Environment Variables

| Variable              | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `BOT_TOKEN`           | Discord bot token                                                    |
| `GUILD_IDS`           | Comma-separated Discord guild IDs                                    |
| `XAI_API_KEY`         | xAI API key for chat/image/video                                     |
| `XAI_COLLECTION_IDS`  | Optional comma-separated collection IDs used by `collections_search` |

If `collections_search=true` and `XAI_COLLECTION_IDS` is empty, converse returns a user-facing error.

## Test Commands

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Notes for Future Changes

- Tool support currently targets exactly four built-in tools for `/grok converse`.
- If new tools are added:
  - Update `src/util.py` constants/builders
  - Update `resolve_selected_tools()` in `src/xai_api.py`
  - Update tool select options in `src/button_view.py`
  - Update README + this file
  - Add/adjust tests in `tests/test_xai_api.py`, `tests/test_button_view.py`, and `tests/test_util.py`
