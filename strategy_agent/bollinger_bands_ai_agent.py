# strategy_agents/bollinger_bands_ai_agent.py
import logging
import pandas as pd
import pandas_ta as ta # For calculating Bollinger Bands

from .base_ai_strategy_agent import BaseAIStrategyAgent, DEFAULT_AI_STRATEGY_MODEL_NAME

logger = logging.getLogger(__name__)

class BollingerBandsAIStrategyAgent(BaseAIStrategyAgent):
    def __init__(self, gcp_project_id, strategy_signal_topic_id, 
                 ai_model_name=DEFAULT_AI_STRATEGY_MODEL_NAME,
                 bb_length=20, bb_std_dev=2.0):
        super().__init__(agent_name="BollingerBands_AI_Agent", 
                         gcp_project_id=gcp_project_id, 
                         strategy_signal_topic_id=strategy_signal_topic_id,
                         ai_model_name=ai_model_name)
        self.bb_length = int(bb_length)
        self.bb_std_dev = float(bb_std_dev)
        logger.info(f"[{self.agent_name}] Initialized with BB Length: {self.bb_length}, StdDev: {self.bb_std_dev}")

    def _reset_specific_agent_state(self):
        # No specific state beyond historical_candles_df for this simple BB agent
        pass

    def _add_strategy_specific_indicators_to_df(self, df_candles):
        """Adds Bollinger Bands to the DataFrame."""
        if 'close' not in df_candles.columns or len(df_candles) < self.bb_length:
            logger.debug(f"[{self.agent_name}] Not enough data or no 'close' column for Bollinger Bands calculation.")
            return df_candles # Return original if BB can't be calculated

        try:
            # pandas_ta expects column names like 'close', not case sensitive by default but good to be consistent
            bbands = ta.bbands(df_candles['close'], length=self.bb_length, std=self.bb_std_dev)
            if bbands is not None and not bbands.empty:
                # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
                df_candles[f'BB_Lower_{self.bb_length}_{self.bb_std_dev}'] = bbands.iloc[:,0].round(2) # Lower band
                df_candles[f'BB_Mid_{self.bb_length}_{self.bb_std_dev}'] = bbands.iloc[:,1].round(2)   # Middle band (SMA)
                df_candles[f'BB_Upper_{self.bb_length}_{self.bb_std_dev}'] = bbands.iloc[:,2].round(2) # Upper band
                df_candles[f'BB_Bandwidth_{self.bb_length}_{self.bb_std_dev}'] = bbands.iloc[:,3].round(4) # Bandwidth
                df_candles[f'BB_Percent_{self.bb_length}_{self.bb_std_dev}'] = bbands.iloc[:,4].round(4)   # %B
                logger.debug(f"[{self.agent_name}] Bollinger Bands calculated and added to DataFrame.")
            else:
                logger.warning(f"[{self.agent_name}] pandas_ta.bbands returned None or empty DataFrame.")
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error calculating Bollinger Bands: {e}", exc_info=True)
        return df_candles

    def _construct_ai_prompt(self, formatted_candle_data_with_indicators):
        current_close_price = self.historical_candles_df['close'].iloc[-1]
        # Attempt to get last BB values for the prompt context
        bb_lower_col = f'BB_Lower_{self.bb_length}_{self.bb_std_dev}'
        bb_mid_col = f'BB_Mid_{self.bb_length}_{self.bb_std_dev}'
        bb_upper_col = f'BB_Upper_{self.bb_length}_{self.bb_std_dev}'
        
        last_bb_lower = self.historical_candles_df[bb_lower_col].iloc[-1] if bb_lower_col in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[bb_lower_col].iloc[-1]) else "N/A"
        last_bb_mid = self.historical_candles_df[bb_mid_col].iloc[-1] if bb_mid_col in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[bb_mid_col].iloc[-1]) else "N/A"
        last_bb_upper = self.historical_candles_df[bb_upper_col].iloc[-1] if bb_upper_col in self.historical_candles_df.columns and not pd.isna(self.historical_candles_df[bb_upper_col].iloc[-1]) else "N/A"

        prompt = f"""
        You are an AI trading assistant interpreting a Bollinger Bands (BB) strategy for intraday trading of '{self.current_instrument_name_display}' (current price ~{current_close_price:.2f}).
        Bollinger Bands parameters: Length={self.bb_length}, Standard Deviations={self.bb_std_dev}.
        Current approximate BB values: Lower={last_bb_lower}, Middle={last_bb_mid}, Upper={last_bb_upper}.

        Here is the recent candle data including OHLCV and calculated Bollinger Bands (BB_Lower, BB_Mid, BB_Upper, BB_Bandwidth, BB_Percent):
        ```csv
        {formatted_candle_data_with_indicators}
        ```

        Analyze this data considering Bollinger Bands principles:
        1. Price relative to bands: Is price touching/outside upper/lower band (potential reversal or breakout)? Is it near the middle band?
        2. Bandwidth: Are bands contracting (low volatility, potential breakout pending) or expanding (high volatility)?
        3. %B Indicator: What is the current %B value (price position relative to bands)?
        4. "Walking the bands": Is price consistently riding the upper or lower band in a strong trend?
        5. Consider candlestick patterns at band extremes (e.g., reversal candle at upper band).

        Based on your Bollinger Bands-focused analysis, provide a recommendation in JSON format:
        {{
          "signal": "BUY" | "SELL" | "HOLD" | "EXIT_POSITION",
          "confidence": float (0.0 to 1.0),
          "reasoning": "String, your concise reasoning based on BB analysis (e.g., 'Price touched lower BB ({last_bb_lower}) and formed a bullish pin bar, suggesting a mean reversion BUY. Current %B is low.').",
          "bb_observed_condition": "TOUCH_LOWER" | "TOUCH_UPPER" | "BREAKOUT_UP" | "BREAKOUT_DOWN" | "SQUEEZE" | "WALKING_BANDS_UP" | "WALKING_BANDS_DOWN" | "NEAR_MIDDLE" | "INSIDE_BANDS_NEUTRAL" | null,
          "current_bb_percent_b": float (latest %B value if available, else null)
        }}
        If price is simply within the bands without clear signals, suggest HOLD.
        If a strong counter-signal appears while in a position (e.g., price breaks decisively against a mean-reversion trade), suggest EXIT_POSITION.

        JSON Response:
        """
        return prompt

# --- Cloud Function Entry Point for BollingerBandsAIStrategyAgent ---
# (This would be similar to other AI strategy agent CF wrappers, 
#  instantiating BollingerBandsAIStrategyAgent and calling its methods.)
#  Example:
#  from .bollinger_bands_ai_agent import BollingerBandsAIStrategyAgent
#  bollinger_bands_ai_agent_instance = None
#  def bollinger_bands_ai_agent_cf_entrypoint(event, context):
#      global bollinger_bands_ai_agent_instance
#      # ... standard initialization and message processing ...
#      # if candle data:
#      #    signal, conf, reason, price, ai_details = bollinger_bands_ai_agent_instance.process_candle_and_generate_signal(message_data)
#      #    bollinger_bands_ai_agent_instance.publish_generated_signal(signal, conf, reason, price, message_data['timestamp'], ai_details)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("--- Testing BollingerBands AI Strategy Agent Locally ---")
    # --- IMPORTANT: Set GEMINI_API_KEY and GCP_PROJECT_ID in your environment for this test ---
    if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("GCP_PROJECT_ID"):
        logger.error("GEMINI_API_KEY or GCP_PROJECT_ID not set. Local test will likely fail.")
    else:
        try:
            agent = BollingerBandsAIStrategyAgent(
                gcp_project_id=os.environ["GCP_PROJECT_ID"],
                strategy_signal_topic_id="local-test-signals" # Mock topic
            )
            mock_instrument = {
                "token": 12345, "name": "TestStockBB", 
                "tradingsymbol_kite": "TESTBB", "exchange": "NSE"
            }
            agent.update_instrument_focus(mock_instrument)

            # Simulate some candle data
            base_time = pd.to_datetime("2023-02-01 09:15:00")
            num_candles_to_generate = agent.min_candles_for_analysis + agent.bb_length + 5 # Ensure enough for BB calc and analysis
            
            for i in range(num_candles_to_generate):
                candle_time = base_time + pd.Timedelta(minutes=i*1) # 1-min candles
                # Create somewhat realistic price movements for BB
                price_offset = pd.np.sin(i / 10) * 5 + pd.np.random.randn() * 2
                mock_candle = {
                    'instrument_token': 12345, 
                    'timestamp': candle_time.isoformat(),
                    'open': 100 + price_offset - 0.5 + pd.np.random.rand()*0.2, 
                    'high': 100 + price_offset + 1.0 + pd.np.random.rand()*0.5,
                    'low': 100 + price_offset - 1.0 - pd.np.random.rand()*0.5,
                    'close': 100 + price_offset + pd.np.random.randn()*0.5,
                    'volume': 1000 + pd.np.random.randint(0,500)
                }
                # Ensure OHLC logic
                mock_candle['high'] = max(mock_candle['high'], mock_candle['open'], mock_candle['close'])
                mock_candle['low'] = min(mock_candle['low'], mock_candle['open'], mock_candle['close'])

                if i >= agent.min_candles_for_analysis -1 :
                     logger.info(f"\n--- Processing candle {i+1} for potential AI signal (Bollinger Bands) ---")
                     signal, conf, reason, price, ai_details = agent.process_candle_and_generate_signal(mock_candle)
                     if signal: # If a signal was attempted (not None)
                         agent.publish_generated_signal(signal, conf, reason, price, mock_candle['timestamp'], ai_details)
                     time.sleep(0.5) # Avoid rapid API calls if live testing
                else:
                     agent._add_candle_to_history(mock_candle)
                     logger.debug(f"Added BB candle {i+1} to history. Total: {len(agent.historical_candles_df)}")
        except Exception as e:
            logger.error(f"Error during local BollingerBandsAIStrategyAgent test: {e}", exc_info=True)

    logger.info("--- BollingerBands AI Strategy Agent local testing finished ---")
