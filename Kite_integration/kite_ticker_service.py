# kite_integration/kite_ticker_service.py
# (This file would be similar to `kite_ticker_service/main.py (Conceptual)` from a previous response,
# but enhanced for robustness and clarity. Key aspects include:)
# - Reading KITE_API_KEY, KITE_ACCESS_TOKEN from environment.
# - Robust WebSocket connection management with reconnection logic.
# - Subscribing to instrument tokens dynamically (e.g., via a Pub/Sub control topic).
# - Publishing received ticks to a dedicated Pub/Sub topic (e.g., PUBSUB_MARKET_TICKS_TOPIC).
# - Graceful error handling and logging.

import os
from kiteconnect import KiteTicker
from google.cloud import pubsub_v1
import json
import time
import logging
import threading

logger = logging.getLogger(__name__)

# --- Configuration from Environment Variables ---
KITE_API_KEY = os.environ.get("KITE_API_KEY")
KITE_ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN") # CRITICAL: Needs daily refresh mechanism
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
PUBSUB_TICKS_TOPIC_ID = os.environ.get("PUBSUB_MARKET_TICKS_TOPIC", "market-ticks-tradeveaver")
# Topic for receiving instrument subscription updates from MarketSelectionAgent
PUBSUB_INSTRUMENT_SUB_UPDATE_TOPIC_ID = os.environ.get("PUBSUB_KITE_TICKER_CONTROL_TOPIC", "kite-ticker-control-tradeveaver")
# Subscription ID for the above topic (this service will be a subscriber)
PUBSUB_INSTRUMENT_SUB_SUBSCRIPTION_ID = os.environ.get("PUBSUB_KITE_TICKER_CONTROL_SUB", "kite-ticker-control-sub")


# --- Global Variables for the Service ---
_publisher_client = None
_ticks_topic_path = None
_kws_client = None # KiteTicker instance
_subscribed_instrument_tokens = set() # Integer tokens
_subscription_lock = threading.Lock() # For thread-safe access to _subscribed_instrument_tokens
_reconnect_attempts = 0
MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_DELAY_BASE_SEC = 5

def _initialize_pubsub_publisher():
    global _publisher_client, _ticks_topic_path
    if not GCP_PROJECT_ID or not PUBSUB_TICKS_TOPIC_ID:
        logger.error("TickerService: GCP_PROJECT_ID or PUBSUB_TICKS_TOPIC_ID not set. Cannot initialize publisher.")
        return False
    try:
        _publisher_client = pubsub_v1.PublisherClient()
        _ticks_topic_path = _publisher_client.topic_path(GCP_PROJECT_ID, PUBSUB_TICKS_TOPIC_ID)
        logger.info(f"TickerService: Pub/Sub publisher initialized for topic: {_ticks_topic_path}")
        return True
    except Exception as e:
        logger.error(f"TickerService: Failed to initialize Pub/Sub publisher: {e}", exc_info=True)
        return False

def _on_ticks_handler(ws, ticks_data_list):
    logger.debug(f"TickerService: Received ticks: {ticks_data_list}")
    if not _publisher_client or not _ticks_topic_path:
        logger.warning("TickerService: Publisher not ready. Skipping tick publishing.")
        return

    for tick_data in ticks_data_list:
        try:
            # Add a service timestamp for latency tracking if needed
            tick_data['service_received_at'] = time.time()
            # Ensure instrument_token is int (KiteTicker usually provides int)
            if 'instrument_token' in tick_data:
                tick_data['instrument_token'] = int(tick_data['instrument_token'])

            tick_message_bytes = json.dumps(tick_data).encode("utf-8")
            future = _publisher_client.publish(_ticks_topic_path, tick_message_bytes)
            future.result(timeout=5) # Wait for publish with timeout
            logger.debug(f"TickerService: Published tick for {tick_data.get('instrument_token')} to {_ticks_topic_path}")
        except TimeoutError:
            logger.warning(f"TickerService: Timeout publishing tick for {tick_data.get('instrument_token')}")
        except Exception as e:
            logger.error(f"TickerService: Error processing or publishing tick: {e}. Tick: {tick_data}", exc_info=True)


def _update_subscriptions(ws_instance):
    """Subscribes/unsubscribes based on _subscribed_instrument_tokens set."""
    with _subscription_lock:
        if not _subscribed_instrument_tokens:
            logger.info("TickerService: No instruments to subscribe to currently.")
            # Optionally unsubscribe from all if previously subscribed
            # current_subs = ws_instance.subscribed_tokens # Fictional method, check KiteTicker docs
            # if current_subs: ws_instance.unsubscribe(list(current_subs))
            return

        tokens_to_subscribe = list(_subscribed_instrument_tokens)
        logger.info(f"TickerService: Attempting to subscribe to tokens: {tokens_to_subscribe}")
        try:
            ws_instance.subscribe(tokens_to_subscribe)
            # MODE_FULL gives OHLC, depth, LTP etc. MODE_LTP is just LTP. MODE_QUOTE for quote.
            ws_instance.set_mode(ws_instance.MODE_FULL, tokens_to_subscribe)
            logger.info(f"TickerService: Subscribed to {len(tokens_to_subscribe)} tokens. Mode set to FULL.")
        except Exception as e:
            logger.error(f"TickerService: Error during WebSocket subscribe/set_mode: {e}", exc_info=True)


def _on_connect_handler(ws, response):
    global _reconnect_attempts
    logger.info(f"TickerService: Kite Ticker WebSocket connected. Response: {response}")
    _reconnect_attempts = 0 # Reset on successful connection
    _update_subscriptions(ws)

def _on_close_handler(ws, code, reason):
    logger.warning(f"TickerService: Kite Ticker WebSocket closed. Code: {code}, Reason: {reason}")
    # Attempt reconnection strategy
    _attempt_reconnect()

def _on_error_handler(ws, code, reason):
    logger.error(f"TickerService: Kite Ticker WebSocket error. Code: {code}, Reason: {reason}")
    # Errors might also lead to a close event, which will trigger reconnection.
    # If it's a critical error that doesn't close, may need specific handling.

def _on_order_update_handler(ws, order_data):
    logger.info(f"TickerService: Received order update: {order_data}")
    # This can be published to a separate Pub/Sub topic if MainAgent needs real-time order fills.
    # For now, just logging.

def _attempt_reconnect():
    global _kws_client, _reconnect_attempts
    if _kws_client and not _kws_client.is_connected():
        _reconnect_attempts += 1
        if _reconnect_attempts <= MAX_RECONNECT_ATTEMPTS:
            delay = RECONNECT_DELAY_BASE_SEC * (2 ** min(_reconnect_attempts -1, 5)) # Exponential backoff up to a limit
            logger.info(f"TickerService: Attempting to reconnect WebSocket (Attempt {_reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}) in {delay} seconds...")
            time.sleep(delay)
            try:
                if _kws_client: # Check again, might have been stopped
                    _kws_client.connect(threaded=True) # Reconnect in background thread
            except Exception as e:
                logger.error(f"TickerService: Error during WebSocket reconnection attempt: {e}", exc_info=True)
        else:
            logger.critical(f"TickerService: Max WebSocket reconnection attempts ({MAX_RECONNECT_ATTEMPTS}) reached. Service might be non-functional. Manual intervention may be required.")
            # Optionally, could stop the service or trigger an alert.


def _instrument_subscription_message_callback(message):
    """Callback for Pub/Sub messages on the instrument subscription control topic."""
    global _subscribed_instrument_tokens, _kws_client
    try:
        data_str = message.data.decode("utf-8")
        logger.info(f"TickerService: Received instrument subscription control message: {data_str}")
        control_data = json.loads(data_str)
        
        action = control_data.get("action", "replace").lower()
        tokens_from_message = set(map(int, control_data.get("tokens", []))) # Ensure integer tokens

        with _subscription_lock:
            made_changes = False
            if action == "replace":
                if _subscribed_instrument_tokens != tokens_from_message:
                    # Unsubscribe from old tokens not in new set
                    to_unsubscribe = list(_subscribed_instrument_tokens - tokens_from_message)
                    if to_unsubscribe and _kws_client and _kws_client.is_connected():
                        logger.info(f"TickerService: Unsubscribing from: {to_unsubscribe}")
                        _kws_client.unsubscribe(to_unsubscribe)
                    
                    _subscribed_instrument_tokens = tokens_from_message
                    made_changes = True
            elif action == "subscribe": # Add to existing
                newly_added = tokens_from_message - _subscribed_instrument_tokens
                if newly_added:
                    _subscribed_instrument_tokens.update(newly_added)
                    made_changes = True
            elif action == "unsubscribe":
                removed = _subscribed_instrument_tokens.intersection(tokens_from_message)
                if removed:
                    _subscribed_instrument_tokens.difference_update(removed)
                    if _kws_client and _kws_client.is_connected(): # Unsubscribe immediately
                         logger.info(f"TickerService: Unsubscribing from: {list(removed)}")
                         _kws_client.unsubscribe(list(removed))
                    # No need to set made_changes to True to re-trigger full update_subscriptions
                    # as unsubscribe is immediate.
            else:
                logger.warning(f"TickerService: Unknown action '{action}' in control message.")

            if made_changes and _kws_client and _kws_client.is_connected():
                logger.info(f"TickerService: Subscription set changed. New set: {list(_subscribed_instrument_tokens)}. Re-evaluating subscriptions on WebSocket.")
                _update_subscriptions(_kws_client) # Re-subscribe with the new complete set if action was replace/subscribe

        message.ack()
    except Exception as e:
        logger.error(f"TickerService: Error processing instrument subscription control message: {e}", exc_info=True)
        message.nack()


def start_kite_ticker_service_main_loop():
    global _kws_client
    logger.info("TickerService: Starting Kite Ticker Service...")

    if not all([KITE_API_KEY, KITE_ACCESS_TOKEN, GCP_PROJECT_ID, PUBSUB_TICKS_TOPIC_ID, PUBSUB_INSTRUMENT_SUB_UPDATE_TOPIC_ID, PUBSUB_INSTRUMENT_SUB_SUBSCRIPTION_ID]):
        logger.critical("TickerService: One or more critical environment variables are missing. Cannot start.")
        return

    if not _initialize_pubsub_publisher():
        logger.critical("TickerService: Failed to initialize Pub/Sub for ticks. Aborting.")
        return

    # --- Initialize KiteTicker Client ---
    try:
        _kws_client = KiteTicker(KITE_API_KEY, KITE_ACCESS_TOKEN)
        _kws_client.on_ticks = _on_ticks_handler
        _kws_client.on_connect = _on_connect_handler
        _kws_client.on_close = _on_close_handler
        _kws_client.on_error = _on_error_handler
        _kws_client.on_order_update = _on_order_update_handler # Optional
        logger.info("TickerService: KiteTicker client configured.")
    except Exception as e:
        logger.critical(f"TickerService: Failed to initialize KiteTicker client: {e}", exc_info=True)
        return

    # --- Start Pub/Sub Subscriber for Control Messages (in a separate thread) ---
    subscriber_client_control = None
    streaming_pull_future_control = None
    try:
        subscriber_client_control = pubsub_v1.SubscriberClient()
        subscription_path_control = subscriber_client_control.subscription_path(GCP_PROJECT_ID, PUBSUB_INSTRUMENT_SUB_SUBSCRIPTION_ID)
        
        def control_subscriber_thread_target():
            nonlocal streaming_pull_future_control # Allow modification
            try:
                logger.info(f"TickerService: Listening for instrument subscription control messages on {subscription_path_control}...")
                streaming_pull_future_control = subscriber_client_control.subscribe(subscription_path_control, callback=_instrument_subscription_message_callback)
                streaming_pull_future_control.result() # Blocks until error or cancelled
            except Exception as sub_exc: # TimeoutError is a common one on shutdown
                if not (isinstance(sub_exc, TimeoutError) or "cancelled" in str(sub_exc).lower()): # Don't log errors for normal shutdown cancellation
                    logger.error(f"TickerService: Control Pub/Sub subscriber error: {sub_exc}", exc_info=True)
            finally:
                logger.info("TickerService: Control Pub/Sub subscriber thread finished.")

        control_thread = threading.Thread(target=control_subscriber_thread_target, daemon=True)
        control_thread.start()
        logger.info("TickerService: Control message subscriber thread started.")
    except Exception as e:
        logger.error(f"TickerService: Failed to start control Pub/Sub subscriber: {e}", exc_info=True)
        # Continue without dynamic subscriptions if this fails, or make it critical.

    # --- Connect to WebSocket ---
    try:
        _kws_client.connect(threaded=True) # Runs in its own background thread
        logger.info("TickerService: KiteTicker connect() called (threaded).")
    except Exception as e:
        logger.critical(f"TickerService: Failed to initiate KiteTicker connection: {e}", exc_info=True)
        if streaming_pull_future_control: streaming_pull_future_control.cancel()
        if subscriber_client_control: subscriber_client_control.close()
        if _publisher_client: _publisher_client.shutdown()
        return

    # --- Main Service Loop (Keep alive, monitor connection) ---
    try:
        while True:
            if _kws_client and not _kws_client.is_connected():
                logger.warning("TickerService: WebSocket is disconnected in main loop. Attempting reconnect sequence.")
                _attempt_reconnect() # This will try to connect again
            
            # Check if control thread is alive (optional, for very robust services)
            if not control_thread.is_alive() and subscriber_client_control: # If thread died unexpectedly
                 logger.error("TickerService: Control subscriber thread seems to have died. This is unexpected.")
                 # Potentially try to restart it, or log critical alert.
                 # For now, just log.

            time.sleep(15) # Check status periodically
    except KeyboardInterrupt:
        logger.info("TickerService: KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        logger.critical(f"TickerService: Unhandled exception in main loop: {e}", exc_info=True)
    finally:
        logger.info("TickerService: Initiating shutdown sequence...")
        if streaming_pull_future_control:
            logger.info("TickerService: Cancelling control Pub/Sub subscriber future...")
            streaming_pull_future_control.cancel()
            try: streaming_pull_future_control.result(timeout=5) # Wait for it to actually cancel
            except: pass # Ignore errors on cancel result
        if subscriber_client_control:
            logger.info("TickerService: Closing control Pub/Sub subscriber client...")
            subscriber_client_control.close()
        if _kws_client and _kws_client.is_connected():
            logger.info("TickerService: Stopping KiteTicker WebSocket client...")
            _kws_client.stop_async() # Non-blocking stop
            _kws_client.close(1000, "Service shutting down") # Attempt graceful close
        if _publisher_client:
            logger.info("TickerService: Shutting down ticks Pub/Sub publisher client...")
            _publisher_client.shutdown()
        logger.info("TickerService: Shutdown complete.")


if __name__ == "__main__":
    # This service is intended to be run as a long-running process (e.g., Docker container on Cloud Run or GCE).
    # For local testing, ensure all environment variables (KITE_API_KEY, KITE_ACCESS_TOKEN, GCP_PROJECT_ID, topic names) are set.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Starting Kite Ticker Service (Conceptual Main Execution) ---")
    
    if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
        logger.critical("CRITICAL: KITE_API_KEY or KITE_ACCESS_TOKEN not found in environment. Ticker service cannot start.")
    elif not GCP_PROJECT_ID:
        logger.critical("CRITICAL: GCP_PROJECT_ID not found in environment. Ticker service cannot start.")
    else:
        # Example: Manually set initial subscriptions for local testing if control topic isn't immediately used
        # with _subscription_lock:
        #    _subscribed_instrument_tokens = {256265, 260105} # NIFTY 50, NIFTY BANK (integer tokens)
        #    logger.info(f"TickerService Local Test: Initial hardcoded subscriptions: {list(_subscribed_instrument_tokens)}")
        
        start_kite_ticker_service_main_loop()
