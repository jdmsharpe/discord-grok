from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTrackDailyCost:
    """Tests for the _track_daily_cost method."""

    @pytest.fixture
    def cog(self, mock_bot):
        with patch("xai_sdk.AsyncClient"):
            from discord_grok import GrokCog

            return GrokCog(bot=mock_bot)

    def test_track_daily_cost_accumulates(self, cog):
        daily = cog._track_daily_cost(1, 3.00)
        assert daily == pytest.approx(3.00)
        daily = cog._track_daily_cost(1, 15.00)
        assert daily == pytest.approx(18.00)

    def test_track_daily_cost_isolates_users(self, cog):
        cog._track_daily_cost(1, 5.00)
        daily = cog._track_daily_cost(2, 3.00)
        assert daily == pytest.approx(3.00)


class TestPruneRuntimeState:
    """Tests for prune_runtime_state — TTL eviction, overflow cap, cascade cleanup."""

    @pytest.fixture
    def cog(self, mock_bot):
        with patch("xai_sdk.AsyncClient"):
            from discord_grok import GrokCog

            cog = GrokCog(bot=mock_bot)
            # cleanup_conversation_files does HTTP; stub it out for state tests.
            cog._cleanup_conversation_files = AsyncMock()
            return cog

    def _make_conversation(self, *, starter=None, age: timedelta = timedelta(0)):
        from discord_grok.cogs.grok.models import (
            ChatCompletionParameters,
            Conversation,
        )

        params = ChatCompletionParameters(
            model="grok-4.20",
            conversation_starter=starter,
        )
        conversation = Conversation(params=params)
        conversation.updated_at = datetime.now(timezone.utc) - age
        return conversation

    async def test_drops_conversations_older_than_ttl(self, cog):
        from discord_grok.cogs.grok.state import CONVERSATION_TTL, prune_runtime_state

        user = MagicMock(spec=["id"])
        user.id = 42
        cog.conversations[1] = self._make_conversation(starter=user, age=CONVERSATION_TTL * 2)
        cog.conversations[2] = self._make_conversation(starter=user, age=timedelta(minutes=5))

        with patch(
            "discord_grok.cogs.grok.state.cleanup_conversation_files",
            new=AsyncMock(),
        ):
            await prune_runtime_state(cog)

        assert 1 not in cog.conversations
        assert 2 in cog.conversations

    async def test_enforces_max_active_cap_by_dropping_oldest(self, cog):
        from discord_grok.cogs.grok import state as state_mod
        from discord_grok.cogs.grok.state import prune_runtime_state

        with patch.object(state_mod, "MAX_ACTIVE_CONVERSATIONS", 2):
            for i in range(4):
                cog.conversations[i] = self._make_conversation(age=timedelta(minutes=i))

            with patch(
                "discord_grok.cogs.grok.state.cleanup_conversation_files",
                new=AsyncMock(),
            ):
                await prune_runtime_state(cog)

            assert len(cog.conversations) == 2
            # Newest-updated survive: ages 0 and 1 (smallest subtracted delta).
            assert {0, 1} == set(cog.conversations)

    async def test_cascade_cleans_orphaned_views(self, cog):
        from discord_grok.cogs.grok.state import CONVERSATION_TTL, prune_runtime_state

        user = MagicMock(spec=["id"])
        user.id = 99
        cog.conversations[5] = self._make_conversation(starter=user, age=CONVERSATION_TTL * 2)

        orphan_view = MagicMock()
        orphan_view.conversation_id = 5
        cog.views[user] = orphan_view

        with patch(
            "discord_grok.cogs.grok.state.cleanup_conversation_files",
            new=AsyncMock(),
        ):
            await prune_runtime_state(cog)

        assert user not in cog.views

    async def test_prunes_daily_costs_older_than_retention(self, cog):
        from discord_grok.cogs.grok.state import (
            DAILY_COST_RETENTION_DAYS,
            prune_runtime_state,
        )

        old_date = (
            datetime.now(timezone.utc) - timedelta(days=DAILY_COST_RETENTION_DAYS + 2)
        ).date()
        fresh_date = datetime.now(timezone.utc).date()
        cog.daily_costs[(1, old_date.isoformat())] = (10.0, datetime.now(timezone.utc))
        cog.daily_costs[(1, fresh_date.isoformat())] = (5.0, datetime.now(timezone.utc))

        with patch(
            "discord_grok.cogs.grok.state.cleanup_conversation_files",
            new=AsyncMock(),
        ):
            await prune_runtime_state(cog)

        assert (1, old_date.isoformat()) not in cog.daily_costs
        assert (1, fresh_date.isoformat()) in cog.daily_costs


class TestConversationTouch:
    """Conversation.touch() updates updated_at to now."""

    def test_touch_advances_updated_at(self):
        from discord_grok.cogs.grok.models import (
            ChatCompletionParameters,
            Conversation,
        )

        conv = Conversation(params=ChatCompletionParameters(model="grok-4.20"))
        original = conv.updated_at
        # Force a measurable gap.
        conv.updated_at = original - timedelta(hours=1)
        conv.touch()
        assert conv.updated_at > original - timedelta(seconds=1)
