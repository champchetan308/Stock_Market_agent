# strategy_agents/macd_ai_agent.py
import logging
import pandas as pd
import pandas_ta as ta

from .base_ai_strategy_agent import BaseAIStrategyAgent, DEFAULT_AI_STRATEGY_MODEL_NAME

logger = logging.getLogger(__name__)

class MACDAIStrategyAgent(BaseAIStrategyAgent):
    def __init__(self, gcp_project_id, strategy_signal_topic_id, 
                 ai_model_name=DEFAULT_AI_STRATEGY_MODEL_NAME,
                 fast_period=12, slow_period=26, signal_period=9,
                 candle_history_count=50, # MACD needs more history
                 min_candles_for_analysis=35):
        super().__init__(agent_name="MACD_AI_Agent", 
                         gcp_project_id=gcp_project_id, 
                         strategy_signal_topic_id=strategy_signal_topic_id,
                         ai_model_name=ai_model_name,
                         candle_history_count=candle_history_count,
                         min_candles_for_analysis=min_candles_for_analysis)
        self.fast_period = int(fast_period)
        self.slow_period = int(slow_period)
        self.signal_period = int(signal_period)
        logger.info(f"[{self.agent_name}] Initialized with MACD Periods: Fast={self.fast_period}, Slow={self.slow_period}, Signal={self.signal_period}")

    def _reset_specific_agent_state(self):
        pass

    def _add_strategy_specific_indicators_to_df(self, df_candles):
        if 'close' not in df_candles.columns or len(df_candles) < self.slow_period + self.signal_period: # Approx min length
            logger.debug(f"[{self.agent_name}] Not enough data for MACD calculation.")
            return df_candles

        try:
            macd_df = ta.macd(df_candles['close'], fast=self.fast_period, slow=self.slow_period, signal=self.signal_period)
            if macd_df is not None and not macd_df.empty:
                # Columns are typically MACD_12_26_9, MACDh_12_26_9 (histogram), MACDs_12_26_9 (signal line)
                df_candles[f'MACD_F{self.fast_period}_S{self.slow_period}'] = macd_df.iloc[:,0].round(4) # MACD Line
                df_candles[f'MACD_Hist_F{self.fast_period}_S{self.slow_period}_Sig{self.signal_period}'] = macd_df.iloc[:,1].round(4) # Histogram
                df_candles[f'MACD_Signal_Sig{self.signal_period}'] = macd_df.iloc[:,2].round(4) # Signal Line
                logger.debug(f"[{self.agent_name}] MACD indicators calculated and added.")
            else:
                logger.warning(f"[{self.agent_name}] pandas_ta.macd returned None or empty.")
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error calculating MACD: {e}", exc_info=True)
        return df_candles

    def _construct_ai_prompt(self, formatted_candle_data_with_indicators):
        current_close_price = self.historical_candles_df['close'].iloc[-1]
        instrument_display = self.current_instrument_name_display or "the current instrument"
        
        # Get latest MACD values for prompt context
        macd_line_col = f'MACD_F{self.fast_period}_S{self.slow_period}'
        macd_hist_col = f'MACD_Hist_F{self.fast_period}_S{self.slow_period}_Sig{self.signal_period}'
        macd_signal_col = f'MACD_Signal_Sig{self.signal_period}'

        last_macd_line = self.historical_candles_df[macd_line_col].iloc[-1] if macd_line_col in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[macd_line_col].iloc[-1]) else "N/A"
        last_macd_hist = self.historical_candles_df[macd_hist_col].iloc[-1] if macd_hist_col in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[macd_hist_col].iloc[-1]) else "N/A"
        last_macd_signal = self.historical_candles_df[macd_signal_col].iloc[-1] if macd_signal_col in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[macd_signal_col].iloc[-1]) else "N/A"

        prompt = f"""
        You are an AI trading assistant interpreting a MACD (Moving Average Convergence Divergence) strategy for intraday trading of '{instrument_display}' (current price ~{current_close_price:.2f}).
        MACD parameters: Fast EMA={self.fast_period}, Slow EMA={self.slow_period}, Signal Line EMA={self.signal_period}.
        Latest MACD values: MACD Line ~{last_macd_line}, Histogram ~{last_macd_hist}, Signal Line ~{last_macd_signal}.

        Recent candle data (Timestamp,Open,High,Low,Close,Volume,MACD_Line,MACD_Histogram,MACD_Signal_Line):
        ```csv
        {formatted_candle_data_with_indicators}
        ```

        Analyze this data based on MACD principles for an intraday context:
        1. MACD Line vs. Signal Line: Has there been a recent crossover? Is it bullish (MACD above Signal) or bearish (MACD below Signal)?
        2. Histogram: Is it positive or negative? Is its magnitude increasing or decreasing (momentum)?
        3. Zero Line Crossovers: Has the MACD line crossed above or below the zero line (indicating longer-term momentum shift)?
        4. Divergence: Any bullish or bearish divergence between price and the MACD Line or Histogram?
        5. Strength of signals: Consider how far the lines are from zero or from each other.

        Based on your MACD-focused analysis, provide a recommendation in JSON format:
        {{
          "signal": "BUY" | "SELL" | "HOLD" | "EXIT_POSITION",
          "confidence": float (0.0 to 1.0),
          "reasoning": "String, your concise reasoning based on MACD analysis (e.g., 'MACD line (current: {last_macd_line}) has just crossed above the signal line (current: {last_macd_signal}), and histogram turned positive, indicating bullish momentum.').",
          "macd_line_value": {last_macd_line if last_macd_line != "N/A" else "null"},
          "macd_histogram_value": {last_macd_hist if last_macd_hist != "N/A" else "null"},
          "macd_signal_line_value": {last_macd_signal if last_macd_signal != "N/A" else "null"},
          "observed_macd_condition": "BULLISH_CROSSOVER" | "BEARISH_CROSSOVER" | "MACD_ABOVE_SIGNAL_POSITIVE_HIST" | "MACD_BELOW_SIGNAL_NEGATIVE_HIST" | "ZERO_LINE_CROSS_UP" | "ZERO_LINE_CROSS_DOWN" | "BULLISH_DIVERGENCE" | "BEARISH_DIVERGENCE" | "NEUTRAL" | null
        }}
        If signals are weak or conflicting, suggest HOLD. If a strong counter-signal appears while in a position, suggest EXIT_POSITION.

        JSON Response:
        """
        return prompt
