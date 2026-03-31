from unittest.mock import patch

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
