# Discord Grok Bot

![Hits](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fjdmsharpe%2Fdiscord-grok%2F&label=discord-grok&icon=github&color=%23198754&message=&style=flat&tz=UTC)
[![Version](https://img.shields.io/github/v/tag/jdmsharpe/discord-grok?sort=semver&label=version)](https://github.com/jdmsharpe/discord-grok/tags)
[![License](https://img.shields.io/github/license/jdmsharpe/discord-grok?label=license)](./LICENSE)
[![CI](https://github.com/jdmsharpe/discord-grok/actions/workflows/main.yml/badge.svg)](https://github.com/jdmsharpe/discord-grok/actions/workflows/main.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

## Overview
A Discord bot built on Pycord 2.0 that integrates xAI's Grok APIs. It provides stateful multi-turn conversations, image generation, video generation, and text-to-speech accessible through Discord slash commands. Chat uses the xAI Responses API directly via `aiohttp` for automatic billing optimization, while image, video, and file operations use the official xAI Python SDK. All commands are cleanly grouped under the `/grok` namespace.

## Features
- **Multi-turn Conversations:** Persistent conversation history with interactive button controls (regenerate, pause/resume, stop) and automatic prompt caching.
- **Multiple Grok Models:** Choose from Grok 4.20 (Multi-Agent, Reasoning, Non-Reasoning), Grok 4.1 Fast (Reasoning, Non-Reasoning), Grok Code Fast 1, Grok 4 Fast (Reasoning, Non-Reasoning), Grok 4 (0709), and Grok 3 / 3 Mini.
- **Multimodal Input:** Supports text, images (JPEG, PNG), and file attachments via the xAI Files API (PDF, TXT, CSV, code files, up to 48 MB).
- **Reasoning & Research:** Reasoning content is displayed in spoilered embeds. Multi-agent research mode offers configurable agent counts (4 for quick, 16 for deep research).
- **Built-In Tools:** Enable `web_search`, `x_search`, `code_execution`, `collections_search`, and preset-backed `mcp`. Tools can be toggled mid-conversation via a dropdown.
- **Advanced Tool Settings:** `/grok chat` exposes model tuning, X search date/media controls, and web search image analysis.
- **Remote MCP Support:** Connect to trusted remote MCP servers via named presets loaded from env or JSON config. Validation enforces HTTPS and preset-level allow-lists.
- **Resilient Error Handling:** Retries transient xAI HTTP failures, preserves async cancellation, and returns safer user-facing Discord error messages for handled chat and conversation-control failures.
- **Citations & Cost Tracking:** Source citations shown in a dedicated "Sources" embed. Per-request cost and token usage tracking includes reasoning, cached token discounts, tool invocation costs, and TTS character-based costs.
- **Media Generation:**
  - **Images:** Generate or remix images using Grok Imagine Image / Pro. Supports batch generation (up to 10 images) and 13 aspect ratios at 1k or 2k resolutions.
  - **Video:** Generate videos from text or image-to-video with adjustable duration (1–15s), aspect ratios, and resolution (720p/480p).
  - **Text-to-Speech:** 5 expressive voices, 20+ languages (with auto-detection), multiple output codecs (MP3, WAV, PCM, etc.), configurable sample/bit rates, and support for xAI speech tags.

### Chat Model Metadata
Shared chat model metadata lives in `src/discord_grok/cogs/grok/command_options.py`. It defines each slash-visible chat model's id, display name, pricing class, capability flags, and default selection, and it is reused by both `/grok chat` command choices and pricing/capability checks.
The current default chat model is `grok-4.20`.

Current slash-visible chat models:
- `grok-4.20-multi-agent` — Grok 4.20 Multi-Agent (premium)
- `grok-4.20` — Grok 4.20 (premium)
- `grok-4.20-non-reasoning` — Grok 4.20 Non-Reasoning (premium)
- `grok-4-1-fast-reasoning` — Grok 4.1 Fast Reasoning (fast)
- `grok-4-1-fast-non-reasoning` — Grok 4.1 Fast Non-Reasoning (fast)
- `grok-code-fast-1` — Grok Code Fast 1 (code_fast)
- `grok-4-fast-reasoning` — Grok 4 Fast Reasoning (fast)
- `grok-4-fast-non-reasoning` — Grok 4 Fast Non-Reasoning (fast)
- `grok-4-0709` — Grok 4 (0709) (legacy_premium)
- `grok-3-mini` — Grok 3 Mini (mini)
- `grok-3` — Grok 3 (legacy_premium)

To regenerate the model list above from the shared metadata:
```bash
python - <<'PY'
import sys

sys.path.insert(0, "src")

from discord_grok.cogs.grok.command_options import generate_model_markdown_lines

print("\n".join(generate_model_markdown_lines()))
PY
```

## Commands

### `/grok chat`
Start a stateful, multi-turn conversation with Grok.
* **Core Inputs:** `prompt`, `system_prompt`, `model`, `attachment`, tool toggles, and optional MCP preset names.
* **Model Tuning:** Adjust `max_tokens`, `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `reasoning_effort`, and `agent_count`.
* **Tool Refinements:** Configure `x_search_images`, `x_search_videos`, `x_search_date_range`, and `web_search_images`.
* **MCP Integration:** Use `mcp` with comma-separated preset names defined in `XAI_MCP_PRESETS_JSON` or `XAI_MCP_PRESETS_PATH`.

### `/grok image`
Generate images from text prompts, or edit/remix an existing image via attachment.

### `/grok video`
Generate videos from text prompts or transform an image into a video.

### `/grok tts`
Convert text to speech audio (Maximum 15,000 characters per request).

### `/grok check_permissions`
Verify the bot has the necessary permissions in the current channel.

## Setup & Installation

### Prerequisites
- Python 3.10+
- Discord Bot Token
- xAI API Key (get one at the [xAI Console](https://console.x.ai/))

### Installation
1. Clone the repository and navigate to the project directory:
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

### MCP Preset Example
Use either `XAI_MCP_PRESETS_JSON` or `XAI_MCP_PRESETS_PATH` with a JSON object keyed by preset name:

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

Each preset supports:
- `url`: Required HTTPS remote MCP server URL.
- `authorization_env_var`: Optional env var that must be set for the preset to be available.
- `allowed_tools`: Optional allow-list of remote MCP tool names.

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
1. Use `/grok chat` to start a conversation.
2. Use the optional `/grok chat` advanced knobs when you need model tuning or search media controls at conversation start.
3. Type messages in the same channel to continue the conversation.
4. Use interactive buttons below responses to:
   - 🔄 Regenerate the last response
   - ⏯️ Pause/resume the conversation
   - ⏹️ End the conversation
5. **Remote MCP Safety:** `mcp` now accepts preset names, not raw URLs. Define presets only for HTTPS servers you control or trust, and prefer narrow `allowed_tools` lists. Sensitive conversation context may be sent to the configured MCP server if the model chooses to call it.

## Development

### Testing
Tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`). All tests are mocked (no real API calls).
```bash
# Install developer tooling if you have not already
python -m pip install -e ".[dev]"

# Run tests locally
python -m pytest -q

# Run tests in Docker
docker build --build-arg PYTHON_VERSION=3.13 -f Dockerfile.test -t discord-grok-test . 
docker run --rm discord-grok-test python -m pytest -q

# Run linting and type checks in Docker
docker run --rm discord-grok-test sh -lc 'ruff check src tests && ruff format --check src tests && pyright'
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
