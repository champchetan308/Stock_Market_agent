# strategy_agents/ema_ai_agent.py
import logging
import pandas as pd
import pandas_ta as ta

from .base_ai_strategy_agent import BaseAIStrategyAgent, DEFAULT_AI_STRATEGY_MODEL_NAME

logger = logging.getLogger(__name__)

class EMAAIStrategyAgent(BaseAIStrategyAgent):
    def __init__(self, gcp_project_id, strategy_signal_topic_id, 
                 ai_model_name=DEFAULT_AI_STRATEGY_MODEL_NAME,
                 short_ema_period=9, long_ema_period=21,
                 candle_history_count=50, 
                 min_candles_for_analysis=30): # Min candles should be at least long_ema_period
        super().__init__(agent_name="EMA_AI_Agent", 
                         gcp_project_id=gcp_project_id, 
                         strategy_signal_topic_id=strategy_signal_topic_id,
                         ai_model_name=ai_model_name,
                         candle_history_count=candle_history_count,
                         min_candles_for_analysis=max(min_candles_for_analysis, long_ema_period + 5)) # Ensure enough for EMA
        self.short_ema_period = int(short_ema_period)
        self.long_ema_period = int(long_ema_period)
        if self.short_ema_period >= self.long_ema_period:
            raise ValueError("Short EMA period must be less than Long EMA period.")
        logger.info(f"[{self.agent_name}] Initialized with EMA Periods: Short={self.short_ema_period}, Long={self.long_ema_period}")

    def _reset_specific_agent_state(self):
        pass

    def _add_strategy_specific_indicators_to_df(self, df_candles):
        if 'close' not in df_candles.columns or len(df_candles) < self.long_ema_period:
            logger.debug(f"[{self.agent_name}] Not enough data for EMA calculation.")
            return df_candles
        try:
            df_candles[f'EMA_{self.short_ema_period}'] = ta.ema(df_candles['close'], length=self.short_ema_period).round(2)
            df_candles[f'EMA_{self.long_ema_period}'] = ta.ema(df_candles['close'], length=self.long_ema_period).round(2)
            logger.debug(f"[{self.agent_name}] EMAs calculated and added.")
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error calculating EMAs: {e}", exc_info=True)
        return df_candles

    def _construct_ai_prompt(self, formatted_candle_data_with_indicators):
        current_close_price = self.historical_candles_df['close'].iloc[-1]
        instrument_display = self.current_instrument_name_display or "the current instrument"

        ema_short_col = f'EMA_{self.short_ema_period}'
        ema_long_col = f'EMA_{self.long_ema_period}'
        last_ema_short = self.historical_candles_df[ema_short_col].iloc[-1] if ema_short_col in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[ema_short_col].iloc[-1]) else "N/A"
        last_ema_long = self.historical_candles_df[ema_long_col].iloc[-1] if ema_long_col in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[ema_long_col].iloc[-1]) else "N/A"

        prompt = f"""
        You are an AI trading assistant interpreting an EMA (Exponential Moving Average) Crossover strategy for intraday trading of '{instrument_display}' (current price ~{current_close_price:.2f}).
        EMA parameters: Short-term EMA={self.short_ema_period}, Long-term EMA={self.long_ema_period}.
        Latest EMA values: EMA({self.short_ema_period}) ~{last_ema_short}, EMA({self.long_ema_period}) ~{last_ema_long}.

        Recent candle data (Timestamp,Open,High,Low,Close,Volume,EMA_{self.short_ema_period},EMA_{self.long_ema_period}):
        ```csv
        {formatted_candle_data_with_indicators}
        ```

        Analyze this data based on EMA Crossover principles for an intraday context:
        1. Crossover Event: Has the Short EMA recently crossed above (bullish/golden cross) or below (bearish/death cross) the Long EMA?
        2. Current State: Is the Short EMA currently above or below the Long EMA? What is the separation?
        3. Price relative to EMAs: Is the current price above both EMAs (bullish), below both (bearish), or between them (consolidation/uncertain)? Are EMAs acting as dynamic support/resistance?
        4. Slope of EMAs: Are the EMAs sloping upwards (uptrend), downwards (downtrend), or flat (ranging)?

        Based on your EMA Crossover-focused analysis, provide a recommendation in JSON format:
        {{
          "signal": "BUY" | "SELL" | "HOLD" | "EXIT_POSITION",
          "confidence": float (0.0 to 1.0),
          "reasoning": "String, your concise reasoning (e.g., 'Short EMA ({self.short_ema_period}, current: {last_ema_short}) has decisively crossed above Long EMA ({self.long_ema_period}, current: {last_ema_long}), and price is holding above both, indicating a new bullish trend.').",
          "ema_short_value": {last_ema_short if last_ema_short != "N/A" else "null"},
          "ema_long_value": {last_ema_long if last_ema_long != "N/A" else "null"},
          "observed_ema_condition": "BULLISH_CROSSOVER_CONFIRMED" | "BEARISH_CROSSOVER_CONFIRMED" | "SHORT_EMA_ABOVE_LONG_EMA" | "SHORT_EMA_BELOW_LONG_EMA" | "PRICE_TESTING_EMA_SUPPORT" | "PRICE_TESTING_EMA_RESISTANCE" | "EMAS_FLAT_RANGING" | "NEUTRAL" | null
        }}
        If signals are weak, EMAs are intertwined, or price is choppy, suggest HOLD. If a strong counter-crossover occurs while in a position, suggest EXIT_POSITION.

        JSON Response:
        """
        return prompt
