# strategy_agents/chart_pattern_ai_agent.py
import logging
import pandas as pd

from .base_ai_strategy_agent import BaseAIStrategyAgent, DEFAULT_AI_STRATEGY_MODEL_NAME

logger = logging.getLogger(__name__)

class ChartPatternAIStrategyAgent(BaseAIStrategyAgent):
    def __init__(self, gcp_project_id, strategy_signal_topic_id, 
                 ai_model_name=DEFAULT_AI_STRATEGY_MODEL_NAME,
                 candle_history_count=60, # Chart patterns often need more context
                 min_candles_for_analysis=30):
        super().__init__(agent_name="ChartPattern_AI_Agent", 
                         gcp_project_id=gcp_project_id, 
                         strategy_signal_topic_id=strategy_signal_topic_id,
                         ai_model_name=ai_model_name,
                         candle_history_count=candle_history_count,
                         min_candles_for_analysis=min_candles_for_analysis)
        logger.info(f"[{self.agent_name}] Initialized for chart pattern analysis.")

    def _reset_specific_agent_state(self):
        pass

    def _add_strategy_specific_indicators_to_df(self, df_candles):
        # This agent primarily works on raw OHLCV, but could add S/R levels or pivot points if desired.
        # For now, no additional indicators are added by default.
        return df_candles

    def _construct_ai_prompt(self, formatted_candle_data_ohlcv):
        current_close_price = self.historical_candles_df['close'].iloc[-1]
        instrument_display = self.current_instrument_name_display or "the current instrument"

        prompt = f"""
        You are an expert AI technical chart analyst for intraday trading of '{instrument_display}' (current price ~{current_close_price:.2f}).
        Analyze the following recent OHLCV candle data for significant chart patterns and candlestick formations.
        The last row in the data is the most recent candle.

        Candle Data (Timestamp,Open,High,Low,Close,Volume):
        ```csv
        {formatted_candle_data_ohlcv}
        ```

        Your analysis should focus on:
        1. Classical Chart Patterns: Identify any emerging or completed patterns like Head and Shoulders (or Inverse), Double/Triple Tops/Bottoms, Triangles (Ascending, Descending, Symmetrical), Flags, Pennants, Wedges, Channels. Describe the pattern and its implication (bullish/bearish).
        2. Candlestick Patterns: Identify any significant single or multi-candlestick reversal or continuation patterns at key price levels (e.g., Bullish/Bearish Engulfing, Hammer, Shooting Star, Doji, Morning/Evening Star).
        3. Support and Resistance: Identify key horizontal support and resistance levels based on recent price action. Note if price is currently testing or breaking any of these levels.
        4. Trend Lines: Can you identify any clear short-term trend lines or channels? Is price respecting or breaking them?
        5. Volume Confirmation: Does volume confirm the identified patterns or breakouts?

        Based on your comprehensive chart pattern analysis, provide a recommendation in JSON format:
        {{
          "signal": "BUY" | "SELL" | "HOLD" | "EXIT_POSITION",
          "confidence": float (0.0 to 1.0, reflecting conviction based on clarity and strength of patterns),
          "reasoning": "String, your concise reasoning detailing the key patterns/levels observed and their implications (e.g., 'Identified a bullish flag pattern breakout on high volume, with price clearing near-term resistance at {current_close_price + 10}. Previous support at {current_close_price - 5} held.').",
          "primary_pattern_identified": "String name of the most significant pattern (e.g., 'Ascending Triangle Breakout', 'Bearish Engulfing at Resistance', 'None Clear')",
          "key_support_level_observed": float | null,
          "key_resistance_level_observed": float | null,
          "pattern_breakout_confirmed": true | false | null (if applicable to the pattern),
          "volume_confirms_pattern": true | false | null
        }}
        If patterns are unclear, conflicting, or still forming, suggest HOLD. If a pattern completes and signals a reversal against an existing position, suggest EXIT_POSITION.
        Focus on patterns relevant for intraday price movement.

        JSON Response:
        """
        return prompt

