# Discord Grok Bot

[![CI](https://github.com/jdmsharpe/discord-grok/actions/workflows/main.yml/badge.svg)](https://github.com/jdmsharpe/discord-grok/actions/workflows/main.yml)

A Discord bot built on [Pycord 2.0](https://github.com/Pycord-Development/pycord) that integrates xAI's Grok API via the official [xAI SDK](https://github.com/xai-org/sdk-python). It provides conversational AI, image generation, and video generation accessible through Discord slash commands.

## Features

All commands are grouped under `/grok` for clean namespacing.

### Text Generation

- **`/grok converse`**: Have multi-turn conversations with Grok AI models
- Support for multiple Grok models:
  - Grok 4.1 Fast (Reasoning / Non-Reasoning)
  - Grok Code Fast 1
  - Grok 4 Fast (Reasoning / Non-Reasoning)
  - Grok 4 (0709)
  - Grok 3, Grok 3 Mini
  - Grok 2 Vision (1212)
- Persistent conversation history with interactive button controls (regenerate, pause/resume, stop)
- Multimodal support (text + images: JPEG, PNG, GIF, WEBP)
- Reasoning content displayed in spoilered embeds for reasoning-capable models
- Customizable system prompts
- Advanced parameters: temperature, top_p, frequency_penalty, presence_penalty, max_tokens

### Image Generation

- **`/grok image`**: Generate images from text prompts
- Model options:
  - Grok Imagine Image Pro
  - Grok Imagine Image (Default)
  - Grok 2 Image (1212)
- Multiple aspect ratios: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3

### Video Generation

- **`/grok video`**: Generate videos from text prompts
- Customizable aspect ratio (16:9, 9:16, 1:1, 4:3, 3:4)
- Adjustable duration and resolution (720p / 480p)

### Utility

- **`/grok check_permissions`**: Verify the bot has the necessary permissions in the current channel

## Setup

### Prerequisites

- Python 3.12+
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
   BOT_TOKEN=your_discord_bot_token
   GUILD_IDS=your_guild_id_1,your_guild_id_2
   XAI_API_KEY=your_xai_api_key
   ```

### Running the Bot

**Directly:**

```bash
python src/bot.py
```

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

1. Use `/grok converse` to start a conversation with Grok
2. Once a conversation is started, simply type messages in the same channel to continue the conversation
3. Use the interactive buttons:
   - :arrows_counterclockwise: Regenerate the last response
   - :play_pause: Pause/resume the conversation
   - :stop_button: End the conversation

## Requirements

- xai-sdk ~1.6
- py-cord ~2.7
- python-dotenv ~1.2

## License

MIT License - see [LICENSE](LICENSE) for details.
