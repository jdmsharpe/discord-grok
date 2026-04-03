"""Thin repo-local launcher retained for `python src/bot.py`."""

import logging

from discord_grok.bot import main

if __name__ == "__main__":
    # Optional standalone default logging for local execution.
    # Embedding applications should configure logging centrally instead.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
