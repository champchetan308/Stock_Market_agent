# strategy_agents/base_ai_strategy_agent.py (Advanced)
import json
import time
import os
import logging
import pandas as pd
from abc import ABC, abstractmethod
from google.cloud import pubsub_v1

# Assuming common.gemini_utils is accessible
from common.gemini_utils import generate_structured_gemini_response, get_gemini_model_instance

logger = logging.getLogger(__name__)

# --- Default Configuration for Base AI Strategy Agents ---
DEFAULT_CANDLE_HISTORY_FOR_AI = 45 # Number of candles to provide to AI
DEFAULT_MIN_CANDLES_FOR_ANALYSIS = 25
DEFAULT_AI_STRATEGY_MODEL_NAME = "gemini-1.5-pro-latest" # Using Pro as requested

class BaseAIStrategyAgent(ABC):
    def __init__(self, agent_name, gcp_project_id, strategy_signal_topic_id, 
                 ai_model_name=DEFAULT_AI_STRATEGY_MODEL_NAME,
                 candle_history_count=DEFAULT_CANDLE_HISTORY_FOR_AI,
                 min_candles_for_analysis=DEFAULT_MIN_CANDLES_FOR_ANALYSIS):
        self.agent_name = agent_name
        self.gcp_project_id = gcp_project_id
        self.strategy_signal_topic_id = strategy_signal_topic_id
        self.ai_model_name = ai_model_name
        self.candle_history_count = int(candle_history_count)
        self.min_candles_for_analysis = int(min_candles_for_analysis)
        
        self.publisher = None
        self.signal_topic_path = None
        self._initialize_publisher() # Robust publisher init

        self.current_instrument_token = None
        self.current_instrument_symbol = None # Kite's tradingsymbol
        self.current_instrument_exchange = None
        self.current_instrument_name_display = None # User-friendly name
        
        self.historical_candles_df = pd.DataFrame() # Stores OHLCV data
        
        try:
            get_gemini_model_instance(model_name=self.ai_model_name) # Ensures model is configured and accessible
            logger.info(f"[{self.agent_name}] Verified Gemini model '{self.ai_model_name}' accessibility.")
        except Exception as e:
            logger.critical(f"[{self.agent_name}] CRITICAL FAILURE: Cannot access Gemini model '{self.ai_model_name}'. Agent will not function. Error: {e}", exc_info=True)
            raise RuntimeError(f"Gemini model '{self.ai_model_name}' inaccessible for {self.agent_name}") from e

        logger.info(f"[{self.agent_name}] initialized. AI Model: '{self.ai_model_name}', Candle History: {self.candle_history_count}, Min Candles: {self.min_candles_for_analysis}")

    def _initialize_publisher(self):
        if not self.gcp_project_id or not self.strategy_signal_topic_id:
            logger.warning(f"[{self.agent_name}] GCP Project ID or Signal Topic ID not set. Pub/Sub publisher NOT initialized.")
            return
        try:
            self.publisher = pubsub_v1.PublisherClient()
            self.signal_topic_path = self.publisher.topic_path(self.gcp_project_id, self.strategy_signal_topic_id)
            logger.info(f"[{self.agent_name}] Pub/Sub publisher initialized for topic: {self.signal_topic_path}")
        except Exception as e:
            logger.error(f"[{self.agent_name}] Failed to initialize Pub/Sub publisher: {e}", exc_info=True)
            self.publisher = None # Ensure it's None on failure

    def update_instrument_focus(self, instrument_data_dict):
        new_token = int(instrument_data_dict.get("token", 0))
        if new_token == 0: logger.error(f"[{self.agent_name}] Received instrument update with invalid token: {instrument_data_dict}"); return

        if new_token != self.current_instrument_token:
            self.current_instrument_token = new_token
            self.current_instrument_symbol = instrument_data_dict.get("tradingsymbol_kite", instrument_data_dict.get("name"))
            self.current_instrument_exchange = instrument_data_dict.get("exchange")
            self.current_instrument_name_display = instrument_data_dict.get("name", self.current_instrument_symbol)
            logger.info(f"[{self.agent_name}] Instrument focus updated to: {self.current_instrument_name_display} (Token: {self.current_instrument_token}, KiteSymbol: {self.current_instrument_symbol})")
            self.reset_strategy_state() # Reset history and agent-specific state
        else:
            logger.debug(f"[{self.agent_name}] Instrument focus update for same instrument: {self.current_instrument_name_display}.")

    def reset_strategy_state(self):
        self.historical_candles_df = pd.DataFrame()
        self._reset_specific_agent_state() # Hook for subclass specific resets
        logger.info(f"[{self.agent_name}] Full strategy state (candle history, specific state) reset for: {self.current_instrument_name_display or 'N/A'}")

    @abstractmethod
    def _reset_specific_agent_state(self):
        """Abstract method for subclasses to reset their unique state variables."""
        pass

    def _add_candle_to_history(self, candle_data_dict):
        """Adds a new candle to historical data. Expects a dict with OHLCV and timestamp."""
        try:
            required_keys = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(k in candle_data_dict for k in required_keys):
                logger.warning(f"[{self.agent_name}] Incomplete candle data: {candle_data_dict}. Skipping history update.")
                return False

            new_candle = pd.DataFrame([{
                'timestamp': pd.to_datetime(candle_data_dict['timestamp']),
                'open': float(candle_data_dict['open']), 'high': float(candle_data_dict['high']),
                'low': float(candle_data_dict['low']), 'close': float(candle_data_dict['close']),
                'volume': float(candle_data_dict['volume'])
            }]).set_index('timestamp')

            if self.historical_candles_df.empty:
                self.historical_candles_df = new_candle
            else:
                self.historical_candles_df = pd.concat([self.historical_candles_df, new_candle])
                # Remove exact duplicate index entries, keeping the last one
                self.historical_candles_df = self.historical_candles_df[~self.historical_candles_df.index.duplicated(keep='last')]
            
            # Maintain history limit (keep slightly more for rolling calculations if needed by AI prompt prep)
            buffer_factor = 1.5 
            if len(self.historical_candles_df) > self.candle_history_count * buffer_factor:
                self.historical_candles_df = self.historical_candles_df.iloc[-int(self.candle_history_count * buffer_factor):]
            
            logger.debug(f"[{self.agent_name}] Added candle. History size: {len(self.historical_candles_df)}. Last candle time: {candle_data_dict['timestamp']}")
            return True
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"[{self.agent_name}] Error processing or adding candle to history: {e}. Data: {candle_data_dict}", exc_info=True)
            return False

    def _prepare_data_for_ai_prompt(self):
        """Prepares recent candle history in a text format for the Gemini prompt."""
        if len(self.historical_candles_df) < self.min_candles_for_analysis:
            logger.debug(f"[{self.agent_name}] Insufficient candle history ({len(self.historical_candles_df)}/{self.min_candles_for_analysis}) for AI analysis.")
            return None
        
        # Select the most recent N candles for analysis by AI
        data_for_prompt = self.historical_candles_df.iloc[-self.candle_history_count:].copy()
        
        # Allow subclasses to add specific indicators to this DataFrame before string conversion
        data_for_prompt = self._add_strategy_specific_indicators_to_df(data_for_prompt)

        # Convert DataFrame to a string format suitable for the AI prompt (e.g., simplified CSV)
        # Ensure consistent formatting, especially for floats.
        header = "Timestamp,Open,High,Low,Close,Volume" + ("," if not data_for_prompt.columns.difference(['open','high','low','close','volume']).empty else "") + \
                 ",".join(col for col in data_for_prompt.columns if col not in ['open','high','low','close','volume'])
        lines = [header]
        for timestamp_idx, row_series in data_for_prompt.iterrows():
            row_str_values = [timestamp_idx.strftime('%Y-%m-%d %H:%M:%S')] + \
                             [f"{row_series[col]:.2f}" if isinstance(row_series[col], (float, int)) and col != 'volume' else 
                              f"{int(row_series[col])}" if col == 'volume' and pd.notna(row_series[col]) else 
                              str(row_series[col]) for col in data_for_prompt.columns]
            lines.append(",".join(row_str_values))
        
        formatted_data_str = "\n".join(lines)
        logger.debug(f"[{self.agent_name}] Prepared data string for AI prompt (last few lines):\n" + "\n".join(lines[-3:]))
        return formatted_data_str

    @abstractmethod
    def _add_strategy_specific_indicators_to_df(self, df_candles):
        """
        Hook for subclasses to add their specific calculated indicators (e.g., RSI, MACD lines)
        to the DataFrame before it's converted to string for the AI prompt.
        Args:
            df_candles (pd.DataFrame): DataFrame of candles to add indicators to.
        Returns:
            pd.DataFrame: DataFrame with added indicator columns.
        """
        return df_candles # Base implementation returns it unchanged

    @abstractmethod
    def _construct_ai_prompt(self, formatted_candle_data_with_indicators):
        """Constructs the specific prompt for this strategy agent to send to Gemini."""
        pass

    def _parse_ai_json_response(self, ai_json_response):
        """
        Parses the structured JSON response from Gemini for this strategy.
        Expected base structure: {"signal": "BUY/SELL/HOLD/EXIT_POSITION", "confidence": 0.0-1.0, "reasoning": "..."}
        Subclasses can override to parse additional strategy-specific fields from the JSON.
        """
        if not ai_json_response or (isinstance(ai_json_response, dict) and "error" in ai_json_response):
            error_msg = f"AI analysis failed or returned error: {ai_json_response.get('message', 'Unknown AI error') if isinstance(ai_json_response, dict) else 'Empty/Invalid AI response'}"
            logger.error(f"[{self.agent_name}] {error_msg}")
            return "HOLD", 0.05, error_msg # Very low confidence hold

        try:
            signal = str(ai_json_response.get("signal", "HOLD")).upper()
            confidence = float(ai_json_response.get("confidence", 0.2)) # Default low confidence
            reasoning = str(ai_json_response.get("reasoning", "No specific reasoning provided by AI."))
            
            valid_signals = ["BUY", "SELL", "HOLD", "EXIT_POSITION"]
            if signal not in valid_signals:
                logger.warning(f"[{self.agent_name}] AI returned unrecognized signal '{signal}'. Defaulting to HOLD.")
                reasoning = f"AI returned unrecognized signal '{signal}'. Original reasoning: {reasoning}"
                signal = "HOLD"
                confidence = 0.1
            
            # Ensure confidence is within bounds
            confidence = max(0.0, min(1.0, confidence))

            return signal, confidence, reasoning, ai_json_response # Return full JSON for detailed logging/use
            
        except (TypeError, ValueError, AttributeError) as e:
            logger.error(f"[{self.agent_name}] Error parsing AI JSON response fields: {e}. Response: {ai_json_response}", exc_info=True)
            return "HOLD", 0.05, f"Error parsing AI response structure: {e}" , ai_json_response


    def process_candle_and_generate_signal(self, candle_data_dict):
        """
        Main method for agent to process a new candle and generate an AI-driven signal.
        Args:
            candle_data_dict (dict): A dictionary representing a new OHLCV candle.
        Returns:
            tuple: (signal_str, confidence_float, reasoning_str, price_at_signal, full_ai_response_dict)
                   Returns (None, ...) if critical error or no focus.
        """
        if not self.current_instrument_token:
            logger.debug(f"[{self.agent_name}] No instrument focus. Skipping signal generation.")
            return None, 0.0, "Agent has no instrument focus.", None, None

        # Ensure candle is for the focused instrument (usually checked by caller/CF wrapper)
        if int(candle_data_dict.get("instrument_token", 0)) != self.current_instrument_token:
            logger.warning(f"[{self.agent_name}] Received candle for non-focused token {candle_data_dict.get('instrument_token')}. Expected {self.current_instrument_token}.")
            return None, 0.0, "Candle for non-focused instrument.", None, None

        if not self._add_candle_to_history(candle_data_dict):
            # Error adding candle or not a full candle
            return "HOLD", 0.1, "Error processing latest candle data.", candle_data_dict.get('close'), None

        if len(self.historical_candles_df) < self.min_candles_for_analysis:
            reason = f"Insufficient candle history ({len(self.historical_candles_df)}/{self.min_candles_for_analysis})."
            logger.debug(f"[{self.agent_name}] {reason}")
            return "HOLD", 0.2, reason, candle_data_dict.get('close'), None

        formatted_data_for_ai = self._prepare_data_for_ai_prompt()
        if not formatted_data_for_ai:
            reason = "Failed to prepare candle data for AI analysis."
            logger.warning(f"[{self.agent_name}] {reason}")
            return "HOLD", 0.1, reason, candle_data_dict.get('close'), None

        ai_prompt = self._construct_ai_prompt(formatted_data_for_ai)
        if not ai_prompt:
            reason = "Internal error: AI prompt generation failed."
            logger.error(f"[{self.agent_name}] {reason}")
            return "HOLD", 0.05, reason, candle_data_dict.get('close'), None

        logger.debug(f"[{self.agent_name}] Requesting AI signal for {self.current_instrument_name_display} using model {self.ai_model_name}.")
        ai_json_response = generate_structured_gemini_response(ai_prompt, model_name=self.ai_model_name)
        
        signal, confidence, reasoning, full_ai_response = self._parse_ai_json_response(ai_json_response)
        
        price_at_signal = candle_data_dict.get('close') # Signal based on close of the processed candle

        logger.info(f"[{self.agent_name}] AI Signal for {self.current_instrument_name_display}: {signal}, Conf: {confidence:.2f}, Price: {price_at_signal:.2f if price_at_signal is not None else 'N/A'}.")
        logger.debug(f"[{self.agent_name}] AI Reasoning: {reasoning[:150]}...")
        
        return signal, confidence, reasoning, price_at_signal, full_ai_response


    def publish_generated_signal(self, signal, confidence, reasoning, price_at_signal, candle_timestamp, full_ai_response=None):
        """Publishes the generated signal to the main agent via Pub/Sub."""
        if not self.publisher or not self.signal_topic_path:
            logger.error(f"[{self.agent_name}] Pub/Sub publisher not configured. Cannot publish signal for {self.current_instrument_name_display}.")
            return

        if not signal or price_at_signal is None:
            logger.warning(f"[{self.agent_name}] Attempted to publish invalid signal (Signal: {signal}, Price: {price_at_signal}). Aborting.")
            return

        payload = {
            "agent_name": self.agent_name,
            "instrument_token": self.current_instrument_token,
            "instrument_symbol": self.current_instrument_symbol, # Kite's tradingsymbol
            "instrument_name_display": self.current_instrument_name_display,
            "exchange": self.current_instrument_exchange,
            "signal": signal.upper(),
            "confidence": round(float(confidence), 4),
            "price_at_signal": round(float(price_at_signal), 2),
            "reasoning": str(reasoning),
            "ai_analysis_details": full_ai_response if isinstance(full_ai_response, dict) else {"raw_response": str(full_ai_response)}, # Store full AI output
            "signal_timestamp": pd.to_datetime(candle_timestamp).timestamp(), # Convert to UNIX timestamp
            "publish_timestamp": time.time()
        }
        message_bytes = json.dumps(payload).encode("utf-8")

        try:
            future = self.publisher.publish(self.signal_topic_path, message_bytes)
            future.result(timeout=10) # Wait for publish confirmation
            logger.info(f"[{self.agent_name}] Published signal for {self.current_instrument_name_display}: {payload['signal']} "
                        f"(Conf: {payload['confidence']:.2f}) to {self.signal_topic_path}")
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error publishing signal for {self.current_instrument_name_display}: {e}", exc_info=True)

    def shutdown(self):
        if self.publisher:
            logger.info(f"[{self.agent_name}] Shutting down Pub/Sub publisher.")
            self.publisher.shutdown()
