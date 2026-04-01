# Discord Grok Bot

![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fjdmsharpe%2Fdiscord-grok%2F&label=discord-grok&icon=github&color=%23198754&message=&style=flat&tz=UTC)
[![CI](https://github.com/jdmsharpe/discord-grok/actions/workflows/main.yml/badge.svg)](https://github.com/jdmsharpe/discord-grok/actions/workflows/main.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

A Discord bot built on [Pycord 2.0](https://github.com/Pycord-Development/pycord) that integrates xAI's Grok APIs. Chat uses the [xAI Responses API](https://docs.x.ai/docs/guides/responses-api) directly via aiohttp for stateful multi-turn conversations with automatic billing optimization. Image, video, and file operations use the official [xAI SDK](https://github.com/xai-org/sdk-python). It provides conversational AI, image generation, video generation, and text-to-speech accessible through Discord slash commands.

## Features

All commands are grouped under `/grok` for clean namespacing.

### Text Generation

- **`/grok chat`**: Have multi-turn conversations with Grok AI models
- Support for multiple Grok models:
  - Grok 4.20 Multi-Agent
  - Grok 4.20 (Reasoning (Default) / Non-Reasoning)
  - Grok 4.1 Fast (Reasoning / Non-Reasoning)
  - Grok Code Fast 1
  - Grok 4 Fast (Reasoning / Non-Reasoning)
  - Grok 4 (0709)
  - Grok 3, Grok 3 Mini
- Persistent conversation history with interactive button controls (regenerate, pause/resume, stop)
- Multimodal support (text + images: JPEG, PNG)
- File attachment support via xAI Files API (PDF, TXT, CSV, code files, etc., up to 48 MB)
- Reasoning content displayed in spoilered embeds for reasoning-capable models
- Customizable system prompts
- Multi-agent research mode with configurable agent count (4 for quick, 16 for deep research)
- Advanced parameters: temperature, top_p, frequency_penalty, presence_penalty, reasoning_effort, agent_count, max_tokens
- Built-in tool calling support for:
  - `web_search`
  - `x_search`
  - `code_execution`
  - `collections_search` (requires `XAI_COLLECTION_IDS`)
- Optional remote MCP support on `/grok chat` through `mcp` and `mcp_allowed_tools`
- Conversation tool toggle dropdown to enable/disable tools mid-conversation
- MCP configuration persists across follow-up turns and is intentionally kept separate from the built-in tool dropdown
- Tool configuration options:
  - X search: date range filter, allowed/excluded handles, image and video understanding
  - Web search: allowed/excluded domains, image understanding
- Remote MCP validation: HTTPS required, hostname-derived label, deduplicated allow-list, and a max of 20 allowed tool names
- Source citations shown in a dedicated "Sources" embed when available
- Per-request cost and token usage tracking with daily cumulative cost per user (includes reasoning, cached, and image token breakdowns, cached token discounts, tool invocation costs, and TTS character-based costs)
- Server-side tool usage counts and invocation costs shown in cost embed when tools are used
- Persistent cost logging via Python logger at every API call site (chat, image, video, TTS)
- Stateful multi-turn conversations via xAI Responses API (`previous_response_id`) with server-side conversation storage and automatic prompt caching

### Image Generation

- **`/grok image`**: Generate images from text prompts, or edit/remix an existing image via attachment
- Model options:
  - Grok Imagine Image Pro (Default)
  - Grok Imagine Image
- Batch generation: generate up to 10 images in a single request via `count` parameter
- 13 aspect ratios: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 2:1, 1:2, 20:9, 9:20, 19.5:9, 9:19.5
- Resolution options: 1k (default), 2k

### Video Generation

- **`/grok video`**: Generate videos from text prompts, or from an image (image-to-video)
- Customizable aspect ratio (16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3)
- Adjustable duration (1–15 seconds) and resolution (720p / 480p)

### Text-to-Speech

- **`/grok tts`**: Convert text to speech audio
- Five expressive voices: Eve, Ara, Rex, Sal, Leo
- 20+ supported languages via BCP-47 codes, plus automatic language detection (`auto`, default)
- Output codecs: MP3, WAV, PCM, μ-law, A-law
- Configurable sample rate (8–48 kHz) and bit rate (32–192 kbps, MP3 only)
- Supports xAI speech tags for expressive delivery (pauses, whispers, emphasis, etc.)
- Maximum 15,000 characters per request

### Utility

- **`/grok check_permissions`**: Verify the bot has the necessary permissions in the current channel

## Setup

### Prerequisites

- Python 3.10+
- Discord Bot Token
- xAI API Key (get one at [xAI Console](https://console.x.ai/))

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jdmsharpe/discord-grok.git
   cd discord-grok
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy the environment example file and fill in your values:

   ```bash
   cp .env.example .env
   ```

5. Edit `.env` with your credentials:

   ```ini
   # Required: Discord bot token for your application
   BOT_TOKEN=your_discord_bot_token

   # Required: Comma-separated Discord guild IDs where slash commands should be registered
   GUILD_IDS=your_guild_id_1,your_guild_id_2

   # Required: xAI API key used for Grok requests
   XAI_API_KEY=your_xai_api_key

   # Optional: Comma-separated collection IDs used by /grok chat collections_search
   # Set this to enable the built-in collections search tool for chat requests
   # XAI_COLLECTION_IDS=collection_id_1,collection_id_2

   # Optional: Show cost and token usage embeds on Grok responses (default: true; accepts true/1/yes)
   # Set this to false to hide pricing and usage details in Discord
   # SHOW_COST_EMBEDS=true
   ```

### Running the Bot

**Directly:**

```bash
python src/bot.py
```

`src/bot.py` remains a thin repo-local launcher that delegates to `discord_grok.bot.main`.
`BOT_TOKEN` and `XAI_API_KEY` are required; the bot exits at startup with a clear error if either is missing or blank.

### Using as a Cog

To compose this repo into a larger bot, import the namespaced package:

```python
from discord_grok import GrokCog

bot.add_cog(GrokCog(bot=bot))
```

Only `src/bot.py` remains at the repository root as a thin launcher; package code should be imported from `discord_grok`.

**With Docker:**

```bash
docker-compose up -d
```

## Discord Bot Setup

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to the "Bot" section and create a bot
4. Enable the following Privileged Gateway Intents:
   - Server Members Intent
   - Message Content Intent
5. Copy the bot token and add it to your `.env` file
6. Go to OAuth2 > URL Generator
7. Select scopes: `bot`, `applications.commands`
8. Select permissions: `Send Messages`, `Read Message History`, `Use Slash Commands`, `Embed Links`, `Attach Files`
9. Use the generated URL to invite the bot to your server

## Usage

1. Use `/grok chat` to start a conversation with Grok
2. Once a conversation is started, simply type messages in the same channel to continue the conversation
3. Use the interactive buttons:
   - :arrows_counterclockwise: Regenerate the last response
   - :play_or_pause_button: Pause/resume the conversation
   - :stop_button: End the conversation
4. Optional MCP settings on `/grok chat`:
   - `mcp`: a single trusted HTTPS remote MCP server URL
   - `mcp_allowed_tools`: optional comma-separated allow-list for that server
5. Built-in tool dropdown changes do not remove an active MCP server; MCP stays attached until the conversation ends

## Remote MCP Safety

`/grok chat` accepts raw remote MCP URLs, so treat them as highly trusted:

- Use only HTTPS servers you control or trust.
- Prefer a narrow `mcp_allowed_tools` allow-list instead of exposing every tool on a server.
- xAI MCP support in this bot does not have a Discord approval loop in this implementation.
- Sensitive conversation context may be sent to the configured MCP server if the model chooses to call it.

## Requirements

- aiohttp ~3.13
- py-cord ~2.7
- python-dotenv ~1.2
- xai-sdk ~1.11

## Development

### Testing

Tests use pytest with pytest-asyncio (`asyncio_mode = "auto"`). All tests are mocked — no real API calls.
The suite is organized around the refactored package layout, with focused files such as `tests/test_grok_cog.py`, `tests/test_grok_chat.py`, `tests/test_grok_client.py`, `tests/test_grok_commands.py`, `tests/test_grok_tooling.py`, `tests/test_config_auth.py`, and `tests/test_lazy_imports.py`.
`tests/test_package_import.py` is the package import smoke test, and `tests/support.py` holds shared Grok test helpers. `tests/test_lazy_imports.py` covers the lazy package exports used by `discord_grok` and `discord_grok.cogs.grok`.
The lazy package exports are paired with type-only imports so `pyright src/` can validate public namespaced imports without eagerly importing the full cog modules at runtime.
Import from `discord_grok` directly; legacy top-level shim modules are no longer part of the supported workflow.

GitHub Actions runs the test suite against Python 3.10, 3.11, 3.12, and 3.13.

```bash
# Run tests
.venv/Scripts/python.exe -m pytest -q    # Windows
.venv/bin/python -m pytest -q            # Unix

# Run tests in Docker
docker build --build-arg PYTHON_VERSION=3.13 -f Dockerfile.test -t discord-grok-test . && docker run --rm discord-grok-test
```

### Linting & Type Checking

```bash
ruff check src/ tests/
ruff format src/ tests/
pyright src/
```

After cloning, run `git config core.hooksPath .githooks` to enable the pre-commit hook.
The pre-commit hook prefers a repo-local `.venv` Ruff binary when available and falls back to `PATH`.

## License

MIT License - see [LICENSE](LICENSE) for details.
