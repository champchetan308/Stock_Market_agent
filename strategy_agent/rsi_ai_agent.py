# strategy_agents/rsi_ai_agent.py
import logging
import pandas as pd
import pandas_ta as ta # For calculating RSI

from .base_ai_strategy_agent import BaseAIStrategyAgent, DEFAULT_AI_STRATEGY_MODEL_NAME

logger = logging.getLogger(__name__)

class RSIAIStrategyAgent(BaseAIStrategyAgent):
    def __init__(self, gcp_project_id, strategy_signal_topic_id, 
                 ai_model_name=DEFAULT_AI_STRATEGY_MODEL_NAME,
                 rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                 candle_history_count=45, # Override base if needed
                 min_candles_for_analysis=25):
        super().__init__(agent_name="RSI_AI_Agent", 
                         gcp_project_id=gcp_project_id, 
                         strategy_signal_topic_id=strategy_signal_topic_id,
                         ai_model_name=ai_model_name,
                         candle_history_count=candle_history_count,
                         min_candles_for_analysis=min_candles_for_analysis)
        self.rsi_period = int(rsi_period)
        self.rsi_overbought = float(rsi_overbought)
        self.rsi_oversold = float(rsi_oversold)
        logger.info(f"[{self.agent_name}] Initialized with RSI Period: {self.rsi_period}, "
                    f"OB: {self.rsi_overbought}, OS: {self.rsi_oversold}")

    def _reset_specific_agent_state(self):
        # No RSI-specific state beyond historical_candles_df in this version
        pass

    def _add_strategy_specific_indicators_to_df(self, df_candles):
        """Adds RSI to the DataFrame for the AI prompt."""
        if 'close' not in df_candles.columns or len(df_candles) < self.rsi_period:
            logger.debug(f"[{self.agent_name}] Not enough data or no 'close' column for RSI calculation.")
            return df_candles

        try:
            df_candles[f'RSI_{self.rsi_period}'] = ta.rsi(df_candles['close'], length=self.rsi_period).round(2)
            logger.debug(f"[{self.agent_name}] RSI_{self.rsi_period} calculated and added to DataFrame.")
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error calculating RSI: {e}", exc_info=True)
            # Add an empty column so the prompt structure doesn't break if AI expects it
            df_candles[f'RSI_{self.rsi_period}'] = pd.NA 
        return df_candles

    def _construct_ai_prompt(self, formatted_candle_data_with_indicators):
        current_close_price = self.historical_candles_df['close'].iloc[-1]
        rsi_col_name = f'RSI_{self.rsi_period}'
        current_rsi_val_str = "N/A"
        if rsi_col_name in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[rsi_col_name].iloc[-1]):
            current_rsi_val_str = f"{self.historical_candles_df[rsi_col_name].iloc[-1]:.2f}"
        
        instrument_display = self.current_instrument_name_display or "the current instrument"

        prompt = f"""
        You are an AI trading assistant interpreting an RSI (Relative Strength Index) strategy for intraday trading of '{instrument_display}' (current price ~{current_close_price:.2f}).
        RSI parameters: Period={self.rsi_period}, Overbought > {self.rsi_overbought}, Oversold < {self.rsi_oversold}.
        The latest calculated RSI value is approximately: {current_rsi_val_str}.

        Recent candle data (Timestamp,Open,High,Low,Close,Volume,RSI_{self.rsi_period}):
        ```csv
        {formatted_candle_data_with_indicators}
        ```

        Analyze this data based on RSI principles for an intraday context:
        1. Current RSI level: Is it overbought, oversold, or neutral?
        2. RSI trend: Is RSI rising, falling, or flat? Does this confirm or diverge from price trend?
        3. Divergence: Any bullish or bearish divergence between price and RSI?
        4. Context: Consider price action (e.g., candlestick patterns, support/resistance) near RSI extremes. For example, RSI oversold in an established uptrend is a stronger buy signal.
        5. Momentum: Does RSI show strong momentum in either direction?

        Based on your RSI-focused analysis, provide a recommendation in JSON format:
        {{
          "signal": "BUY" | "SELL" | "HOLD" | "EXIT_POSITION",
          "confidence": float (0.0 to 1.0, e.g., 0.75, reflecting conviction based purely on this RSI analysis),
          "reasoning": "String, your concise reasoning based on RSI analysis (e.g., 'RSI at 25 (oversold) with bullish divergence as price made a lower low but RSI made a higher low, suggesting potential upward reversal. Current RSI is {current_rsi_val_str}.').",
          "rsi_value_observed": {current_rsi_val_str if current_rsi_val_str != "N/A" else "null"},
          "rsi_condition_summary": "OVERBOUGHT" | "OVERSOLD" | "NEUTRAL_RISING" | "NEUTRAL_FALLING" | "NEUTRAL_FLAT" | "BULLISH_DIVERGENCE" | "BEARISH_DIVERGENCE" | "OTHER_OBSERVATION"
        }}
        If RSI is neutral and no clear divergence or strong momentum, signal HOLD.
        If already in a position and RSI shows a strong reversal (e.g., extreme overbought after a BUY, or strong bearish divergence), suggest EXIT_POSITION.
        Your analysis should be strictly based on the provided data and RSI principles.

        JSON Response:
        """
        return prompt

# --- Cloud Function Entry Point for RSIAIStrategyAgent ---
# (Conceptual - to be implemented in a separate file or main execution script)
# rsi_ai_agent_instance = None
# def rsi_ai_agent_cf_entrypoint(event, context):
#     global rsi_ai_agent_instance
#     if rsi_ai_agent_instance is None:
#         rsi_ai_agent_instance = RSIAIStrategyAgent(
#             gcp_project_id=os.environ.get("GCP_PROJECT_ID"),
#             strategy_signal_topic_id=os.environ.get("PUBSUB_STRATEGY_SIGNALS_TOPIC"),
#             # Add other params from env if needed
#         )
#     # ... standard message decoding and calling:
#     # agent_instance.update_instrument_focus(instrument_data)
#     # OR
#     # signal, conf, reason, price, ai_details = agent_instance.process_candle_and_generate_signal(candle_data)
#     # agent_instance.publish_generated_signal(signal, conf, reason, price, candle_data['timestamp'], ai_details)
