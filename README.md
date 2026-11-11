<h1 align="center">Discord Grok Bot</h1>

<div align="center">

![Hits](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fjdmsharpe%2Fdiscord-grok%2F&label=discord-grok&icon=github&color=%23198754&message=&style=flat&tz=UTC)
[![Version](https://img.shields.io/github/v/tag/jdmsharpe/discord-grok?sort=semver&label=version)](https://github.com/jdmsharpe/discord-grok/tags)
[![License](https://img.shields.io/github/license/jdmsharpe/discord-grok?label=license)](./LICENSE)
[![CI](https://github.com/jdmsharpe/discord-grok/actions/workflows/main.yml/badge.svg)](https://github.com/jdmsharpe/discord-grok/actions/workflows/main.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/jsgreen152/discord-grok?logo=docker&logoColor=white)](https://hub.docker.com/r/jsgreen152/discord-grok)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Codecov](https://codecov.io/gh/jdmsharpe/discord-grok/branch/main/graph/badge.svg)](https://app.codecov.io/gh/jdmsharpe/discord-grok)

</div>

## Overview

A Discord bot for xAI's Grok API, built on Pycord 2.0. It provides chat, image and video generation and text-to-speech through slash commands.

## Features

- **Chat:** Multi-turn conversations with Grok 4.7 (default), 4.6, 4.5, 4.3, 4.20 (Multi-Agent, Reasoning, Non-Reasoning) and Grok Build 0.1. Messages you send in the channel continue the conversation.
- **Attachments:** Images (JPEG, PNG) and files through the xAI Files API (PDF, TXT, CSV, code files, up to 48 MB).
- **Reasoning:** Reasoning content appears in a spoiler. Grok 4.20 Multi-Agent runs 4 agents (quick) or 16 (deep research).
- **Built-in tools:** `web_search`, `x_search`, `code_execution` and `collections_search`.
- **Remote MCP:** Trusted HTTPS servers enabled through named presets, with optional tool allow-lists.
- **Citations:** Source citations appear in a Sources embed.
- **Prompt caching:** Conversations are cached automatically.
- **Media generation:**
  - **Images:** Grok Imagine Image 2.0 (default) or Grok Imagine Image, up to 10 images per request, 13 aspect ratios, 1k or 2k.
  - **Video:** Grok Imagine Video 1.5 (Preview, default), 1.5 Lite or Grok Imagine Video, from text or an image, 1 to 15 seconds.
  - **Text-to-speech:** 5 voices, 20+ languages with auto-detection, several output codecs and xAI speech tags.
- **Cost line:** Each reply ends with a line such as `$0.0014 · 1.3k in (1.2k cached) / 107 out (102 thinking) · 1 search · <$0.01 today`: the cost xAI reports for the request, input tokens (cached in parentheses), output tokens (thinking in parentheses), tool calls and your total for the day. Set `SHOW_COST_EMBEDS=false` to hide it.

## Commands

### `/grok chat`

Start a conversation with Grok.

- **`prompt` (required):** Your first message.
- **`system_prompt`:** System prompt.
- **`model`:** Grok model (default: Grok 4.7).
- **`attachment`:** An image or document.
- **`max_tokens` / `temperature` / `top_p`:** Output limit and sampling (`temperature` and `top_p` 0–1).
- **`frequency_penalty` / `presence_penalty`:** Repetition controls (non-reasoning models only).
- **`reasoning_effort`:** `low`, `medium`, `high` or `xhigh` on Grok 4.7, 4.6, 4.5 and 4.3, plus `none` on 4.3.
- **`agent_count`:** 4 (quick) or 16 (deep research) agents on Grok 4.20 Multi-Agent.
- **`service_tier`:** `priority` for faster scheduling. Token and cache prices double only when xAI grants the priority tier; tool fees don't change.
- **`web_search` / `x_search` / `code_execution` / `collections_search`:** Enable built-in tools (default: off). `collections_search` needs `XAI_COLLECTION_IDS`.
- **`x_search_images` / `x_search_videos` / `web_search_images`:** Let the search tools analyze images and videos (default: off).
- **`x_search_date_range`:** Limit X search to `YYYY-MM-DD,YYYY-MM-DD` (start, end).
- **`mcp`:** Comma-separated MCP preset names (see [MCP Setup](#mcp-setup)).

### `/grok-media image`

Generate images from a prompt, or edit an attached image.

- **`model`:** Grok Imagine Image 2.0 (default) or Grok Imagine Image.
- **`aspect_ratio` / `resolution`:** 13 aspect ratios (default `1:1`) at `1k` (default) or `2k`.
- **`quality`:** Generation quality (priced on Image 2.0 only).
- **`count`:** 1 to 10 images (default 1); not available for edits.
- **`attachment`:** An image to edit or remix.

### `/grok-media video`

Generate a video from a prompt, or animate an attached image.

- **`model`:** Grok Imagine Video 1.5 (Preview, default), 1.5 Lite or Grok Imagine Video.
- **`duration`:** 1 to 15 seconds (default 5).
- **`resolution`:** `480p`, `720p` (default) or `1080p`; `1080p` on Video 1.5 and 1.5 Lite only (Lite upscales it from 720p).
- **`aspect_ratio`:** Video aspect ratio (default `16:9`).
- **`attachment`:** An image to use as the first frame.

### `/grok-tools tts`

Convert text to speech (up to 15,000 characters).

- **`voice`:** 5 voices (default: Eve).
- **`language`:** BCP-47 code, or `auto` (default) to detect it.
- **`output_format` / `sample_rate` / `bit_rate`:** Codec (default MP3), sample rate (default 24000 Hz) and MP3 bit rate (default 128000).

### `/grok check_permissions`

Check if the bot has the necessary permissions in the current channel.

## Setup & Installation

### Prerequisites

- Python 3.11+
- Discord Bot Token
- xAI API Key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jdmsharpe/discord-grok.git
   cd discord-grok
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package and its runtime dependencies:

   ```bash
   python -m pip install .
   ```

4. Copy the environment example file:

   ```bash
   cp .env.example .env
   ```

### Contributor Setup

Install development tooling for tests, linting, and type checking:

```bash
python -m pip install -e ".[dev]"
```

### Configuration (`.env`)

| Variable | Required | Description |
| --- | --- | --- |
| `BOT_TOKEN` | **Yes** | Your Discord bot token |
| `GUILD_IDS` | **Yes** | Comma-separated Discord server IDs |
| `XAI_API_KEY` | **Yes** | Your xAI API key |
| `XAI_COLLECTION_IDS` | No | Comma-separated collection IDs for `/grok chat collections_search` |
| `XAI_MCP_PRESETS_JSON` | No | Inline JSON object of named remote MCP presets for `/grok chat` |
| `XAI_MCP_PRESETS_PATH` | No | Path to a JSON file containing named remote MCP presets |
| `SHOW_COST_EMBEDS` | No | Show cost/token usage details in Discord (Default: `true`) |
| `XAI_PRICING_PATH` | No | Path to a pricing YAML that overrides the bundled `src/discord_grok/config/pricing.yaml` |
| `LOG_FORMAT` | No | `text` (default) for human-readable logs, or `json` for structured JSON-lines output with per-request IDs |
| `SAFETY_IDENTIFIER_SECRET` | No | HMAC key for the per-user identifier sent on chat and image requests (default: a key derived from `BOT_TOKEN`) |

#### User Identifier

Chat requests (including follow-ups) and image requests carry a per-user identifier (`safety_identifier` on chat, `user` on the Images API) so xAI can attribute a policy violation to one Discord user instead of the whole bot.

- The identifier is an HMAC-SHA256 of the Discord user ID, keyed with `SAFETY_IDENTIFIER_SECRET` or, when that is unset, a key derived from `BOT_TOKEN`. xAI never receives the user ID or the key, and the bot refuses to start when `SAFETY_IDENTIFIER_SECRET` equals `XAI_API_KEY`.
- Changing the key (or the bot token, when no secret is set) gives every user a new identifier. To keep identifiers stable across token changes, set the secret to a long random string, for example `python -c "import secrets; print(secrets.token_hex(32))"`.
- discord-claude, discord-grok, discord-openai and discord-openrouter compute it the same way, so a user gets the same identifier from each bot that shares the key.
- Video and text-to-speech requests carry none, because those calls have no identifier field.

#### MCP Setup

Configure named presets in `XAI_MCP_PRESETS_JSON` or `XAI_MCP_PRESETS_PATH`. Example:

```json
{
  "trusted_docs": {
    "url": "https://mcp.example.com/sse",
    "allowed_tools": ["search", "browse"]
  },
  "private_ops": {
    "url": "https://ops.example.com/sse",
    "authorization_env_var": "OPS_MCP_TOKEN"
  }
}
```

Each preset takes `url` (required, HTTPS), `authorization_env_var` (optional; the preset is available only when that variable is set) and `allowed_tools` (optional tool allow-list).

Define presets only for servers you trust: the model can send conversation content to any server it calls.

### Running the Bot

**Locally:**

```bash
python src/bot.py
```

*(Note: `src/bot.py` is a thin launcher that delegates to `discord_grok.bot.main`)*

**With Docker:**

```bash
docker compose up -d --build
```

**Using as a Cog:**
To compose this repo into a larger bot, import the namespaced package:

```python
from discord_grok import GrokCog

bot.add_cog(GrokCog(bot=bot))
```

## Discord Bot Setup

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Create a new application and add a bot in the "Bot" section.
3. Enable **Server Members Intent** and **Message Content Intent** under Privileged Gateway Intents.
4. Copy the bot token and add it to your `.env` file.
5. Go to OAuth2 > URL Generator.
6. Select scopes: `bot`, `applications.commands`.
7. Select permissions: `Send Messages`, `Read Message History`, `Use Slash Commands`, `Embed Links`, `Attach Files`.
8. Use the generated URL to invite the bot to your server.

## Usage

1. Run `/grok chat` to start a conversation.
2. Send messages in the same channel to continue it.
3. Use the controls under each reply:
   - 🔄 Regenerate the last response
   - ⏯️ Pause or resume the conversation
   - ⏹️ End the conversation
   - **Tools** menu: turn tools on or off mid-conversation

## Development

### Testing

Tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`). All tests are mocked (no real API calls).

```bash
# Install developer tooling if you have not already
python -m pip install -e ".[dev]"

# Run tests locally
python -m pytest -q

# Run tests in Docker
docker build --build-arg PYTHON_VERSION=3.11 -f Dockerfile.test -t discord-grok-test:3.11 .
docker run --rm discord-grok-test:3.11 python -m pytest -q

# Run linting and type checks in Docker
docker run --rm discord-grok-test:3.11 sh -lc 'ruff check src tests && ruff format --check src tests && pyright'
```

### Linting & Type Checking

```bash
ruff check src tests
ruff format --check src tests
pyright
```

*Run `git config core.hooksPath .githooks` after cloning to enable the pre-commit hook.*

## License

MIT License - see [LICENSE](LICENSE) for details.
