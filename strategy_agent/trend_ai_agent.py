# strategy_agents/trend_ai_agent.py
import logging
import pandas as pd
import pandas_ta as ta # For indicators like SuperTrend, ADX

from .base_ai_strategy_agent import BaseAIStrategyAgent, DEFAULT_AI_STRATEGY_MODEL_NAME

logger = logging.getLogger(__name__)

class TrendAIStrategyAgent(BaseAIStrategyAgent):
    def __init__(self, gcp_project_id, strategy_signal_topic_id, 
                 ai_model_name=DEFAULT_AI_STRATEGY_MODEL_NAME,
                 adx_length=14, supertrend_atr_period=10, supertrend_atr_multiplier=3.0,
                 candle_history_count=60, 
                 min_candles_for_analysis=40): # ADX/Supertrend need some history
        super().__init__(agent_name="Trend_AI_Agent", 
                         gcp_project_id=gcp_project_id, 
                         strategy_signal_topic_id=strategy_signal_topic_id,
                         ai_model_name=ai_model_name,
                         candle_history_count=candle_history_count,
                         min_candles_for_analysis=min_candles_for_analysis)
        self.adx_length = int(adx_length)
        self.supertrend_atr_period = int(supertrend_atr_period)
        self.supertrend_atr_multiplier = float(supertrend_atr_multiplier)
        logger.info(f"[{self.agent_name}] Initialized with ADX Length: {self.adx_length}, SuperTrend ATR: {self.supertrend_atr_period}, Multiplier: {self.supertrend_atr_multiplier}")

    def _reset_specific_agent_state(self):
        pass

    def _add_strategy_specific_indicators_to_df(self, df_candles):
        if 'close' not in df_candles.columns or 'high' not in df_candles.columns or 'low' not in df_candles.columns \
           or len(df_candles) < max(self.adx_length, self.supertrend_atr_period) + 5: # Ensure enough data
            logger.debug(f"[{self.agent_name}] Not enough data for ADX/SuperTrend calculation.")
            return df_candles
        try:
            # ADX
            adx_df = ta.adx(df_candles['high'], df_candles['low'], df_candles['close'], length=self.adx_length)
            if adx_df is not None and not adx_df.empty:
                # ADX_14, DMP_14, DMN_14
                df_candles[f'ADX_{self.adx_length}'] = adx_df.iloc[:,0].round(2)
                df_candles[f'ADX_DI_plus_{self.adx_length}'] = adx_df.iloc[:,1].round(2) # +DI
                df_candles[f'ADX_DI_minus_{self.adx_length}'] = adx_df.iloc[:,2].round(2) # -DI
            
            # SuperTrend
            supertrend_df = ta.supertrend(df_candles['high'], df_candles['low'], df_candles['close'], 
                                          length=self.supertrend_atr_period, multiplier=self.supertrend_atr_multiplier)
            if supertrend_df is not None and not supertrend_df.empty:
                # SUPERT_7_3.0 (SuperTrend line), SUPERTd_7_3.0 (Direction: 1 for uptrend, -1 for downtrend), ...
                df_candles[f'SuperTrend_L{self.supertrend_atr_period}_M{self.supertrend_atr_multiplier}'] = supertrend_df.iloc[:,0].round(2)
                df_candles[f'SuperTrend_Direction_L{self.supertrend_atr_period}_M{self.supertrend_atr_multiplier}'] = supertrend_df.iloc[:,1] # Direction
            
            logger.debug(f"[{self.agent_name}] ADX and SuperTrend indicators calculated.")
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error calculating Trend indicators: {e}", exc_info=True)
        return df_candles

    def _construct_ai_prompt(self, formatted_candle_data_with_indicators):
        current_close_price = self.historical_candles_df['close'].iloc[-1]
        instrument_display = self.current_instrument_name_display or "the current instrument"

        # Get latest ADX/SuperTrend values for prompt context
        adx_col = f'ADX_{self.adx_length}'
        st_dir_col = f'SuperTrend_Direction_L{self.supertrend_atr_period}_M{self.supertrend_atr_multiplier}'
        
        last_adx = self.historical_candles_df[adx_col].iloc[-1] if adx_col in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[adx_col].iloc[-1]) else "N/A"
        last_st_dir_val = self.historical_candles_df[st_dir_col].iloc[-1] if st_dir_col in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[st_dir_col].iloc[-1]) else "N/A"
        last_st_dir_str = "UP" if last_st_dir_val == 1 else "DOWN" if last_st_dir_val == -1 else "NEUTRAL/CHANGING"


        prompt = f"""
        You are an AI trading assistant interpreting Trend-Following strategies for intraday trading of '{instrument_display}' (current price ~{current_close_price:.2f}).
        Key indicators considered: ADX (Length={self.adx_length}) for trend strength, and SuperTrend (ATR Period={self.supertrend_atr_period}, Multiplier={self.supertrend_atr_multiplier}) for trend direction.
        Latest ADX ~{last_adx}. Latest SuperTrend Direction: {last_st_dir_str} (1 means UP, -1 means DOWN).

        Recent candle data (includes OHLCV, ADX, ADX_DI_plus, ADX_DI_minus, SuperTrend_Line, SuperTrend_Direction):
        ```csv
        {formatted_candle_data_with_indicators}
        ```

        Analyze this data based on Trend-Following principles for an intraday context:
        1. Trend Direction: What is the current trend direction indicated by SuperTrend? Is price above or below the SuperTrend line?
        2. Trend Strength (ADX): Is ADX above a certain threshold (e.g., 20-25) indicating a strong trend? Is ADX rising or falling?
        3. DI Crossover (+DI vs -DI): Does the Directional Movement Index (+DI, -DI) confirm the trend direction (e.g., +DI above -DI for uptrend)?
        4. Entry/Continuation: Are there signals to enter a new trend-following position or add to an existing one? (e.g., price pullback to SuperTrend line in an established trend).
        5. Trend Exhaustion/Reversal: Any signs of trend weakening (e.g., falling ADX during a trend, SuperTrend flip)?

        Based on your Trend-focused analysis, provide a recommendation in JSON format:
        {{
          "signal": "BUY" | "SELL" | "HOLD" | "EXIT_POSITION",
          "confidence": float (0.0 to 1.0),
          "reasoning": "String, your concise reasoning (e.g., 'SuperTrend is bullish (direction: {last_st_dir_str}), price is above ST line, and ADX ({last_adx}) is rising above 25, confirming strong uptrend. Consider BUY on pullbacks or continuation.').",
          "trend_direction_supertrend": "{last_st_dir_str}",
          "trend_strength_adx_value": {last_adx if last_adx != "N/A" else "null"},
          "adx_interpretation": "STRONG_TREND" | "WEAK_TREND" | "NO_TREND" | null,
          "observed_trend_condition": "UPTREND_CONFIRMED" | "DOWNTREND_CONFIRMED" | "RANGING_NO_CLEAR_TREND" | "POTENTIAL_TREND_REVERSAL" | null
        }}
        If no clear trend or conflicting signals, suggest HOLD. If trend shows signs of reversal while in a position, suggest EXIT_POSITION.

        JSON Response:
        """
        return prompt
