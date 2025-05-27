# common/utils.py
import logging
import time
from datetime import datetime, timezone
import pytz # For timezone handling, e.g., Asia/Kolkata

logger = logging.getLogger(__name__)

# --- Constants ---
INDIAN_MARKET_TIMEZONE = 'Asia/Kolkata'

# --- Time Related Utilities ---
def get_current_ist_time():
    """Returns the current time in IST (Indian Standard Time)."""
    try:
        ist = pytz.timezone(INDIAN_MARKET_TIMEZONE)
        return datetime.now(ist)
    except pytz.exceptions.UnknownTimeZoneError:
        logger.error(f"Unknown timezone: {INDIAN_MARKET_TIMEZONE}. Falling back to UTC.")
        return datetime.now(timezone.utc)
    except Exception as e:
        logger.error(f"Error getting IST time: {e}", exc_info=True)
        return datetime.now(timezone.utc) # Fallback

def is_market_hours_ist(current_ist_time=None, 
                        market_open_hour=9, market_open_minute=15,
                        market_close_hour=15, market_close_minute=30):
    """
    Checks if the current IST time is within typical Indian stock market hours.
    Note: Does not account for holidays or pre/post market sessions.
    Args:
        current_ist_time (datetime, optional): The current IST time. If None, fetches it.
        market_open_hour (int): Market open hour (24-hour format).
        market_open_minute (int): Market open minute.
        market_close_hour (int): Market close hour.
        market_close_minute (int): Market close minute.
    Returns:
        bool: True if within market hours, False otherwise.
    """
    if current_ist_time is None:
        current_ist_time = get_current_ist_time()

    # Check if it's a weekday (Monday=0, Sunday=6)
    if current_ist_time.weekday() >= 5: # Saturday or Sunday
        logger.debug(f"Market check: It's a weekend ({current_ist_time.strftime('%A')}). Market closed.")
        return False

    market_open = current_ist_time.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
    market_close = current_ist_time.replace(hour=market_close_hour, minute=market_close_minute, second=0, microsecond=0)

    is_open = market_open <= current_ist_time < market_close
    logger.debug(f"Market hours check: Current IST: {current_ist_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                 f"Open: {market_open.strftime('%H:%M')}, Close: {market_close.strftime('%H:%M')}. Is Open: {is_open}")
    return is_open


# --- Calculation Utilities ---
def calculate_percentage_change(old_value, new_value):
    """Calculates percentage change from old_value to new_value."""
    if old_value is None or new_value is None or old_value == 0:
        return 0.0  # Avoid division by zero or issues with None
    try:
        old_val_f = float(old_value)
        new_val_f = float(new_value)
        if old_val_f == 0: return float('inf') if new_val_f > 0 else float('-inf') if new_val_f < 0 else 0.0
        return ((new_val_f - old_val_f) / abs(old_val_f)) * 100.0
    except (ValueError, TypeError):
        logger.warning(f"Could not calculate percentage change for values: {old_value}, {new_value}")
        return 0.0

def calculate_target_price(entry_price, percentage, side="BUY"):
    """Calculates a target price based on entry and percentage."""
    if entry_price is None or percentage is None: return None
    try:
        entry_p = float(entry_price)
        pct = float(percentage) / 100.0 # Convert percentage to decimal
        if side.upper() == "BUY":
            return entry_p * (1 + pct)
        elif side.upper() == "SELL":
            return entry_p * (1 - pct)
        else:
            logger.warning(f"Invalid side '{side}' for calculate_target_price.")
            return None
    except (ValueError, TypeError):
        logger.warning(f"Could not calculate target price for entry: {entry_price}, pct: {percentage}")
        return None

# --- String/Data Utilities ---
def truncate_string(text, max_length=100, suffix="..."):
    """Truncates a string to a maximum length, adding a suffix if truncated."""
    if not isinstance(text, str) or len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# --- Retry Decorator (Optional, for functions not using KiteHelper's retry or Gemini's retry) ---
def retry_on_exception(max_attempts=3, delay_seconds=1, exceptions_to_catch=(Exception,)):
    """
    A decorator to retry a function call on specified exceptions.
    Args:
        max_attempts (int): Maximum number of attempts.
        delay_seconds (int): Initial delay between retries (exponential backoff applied).
        exceptions_to_catch (tuple): Tuple of exception types to catch and retry on.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions_to_catch as e:
                    attempts += 1
                    logger.warning(f"Retryable error in {func.__name__} (Attempt {attempts}/{max_attempts}): {type(e).__name__} - {e}")
                    if attempts >= max_attempts:
                        logger.error(f"Max retries reached for {func.__name__}. Last error: {e}", exc_info=True)
                        raise # Re-raise the last exception
                    current_delay = delay_seconds * (2 ** (attempts - 1))
                    logger.info(f"Retrying {func.__name__} in {current_delay} seconds...")
                    time.sleep(current_delay)
        return wrapper
    return decorator


# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    logger.info(f"Current IST Time: {get_current_ist_time().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Is market open now (9:15-15:30 IST)? {is_market_hours_ist()}")
    
    # Test on a specific time
    test_time_open = get_current_ist_time().replace(hour=10, minute=0)
    logger.info(f"Is market open at {test_time_open.strftime('%H:%M')} IST? {is_market_hours_ist(current_ist_time=test_time_open)}")
    test_time_closed = get_current_ist_time().replace(hour=16, minute=0)
    logger.info(f"Is market open at {test_time_closed.strftime('%H:%M')} IST? {is_market_hours_ist(current_ist_time=test_time_closed)}")
    test_time_weekend = datetime(2024, 5, 25, 11, 0, 0, tzinfo=pytz.timezone(INDIAN_MARKET_TIMEZONE)) # A Saturday
    logger.info(f"Is market open on {test_time_weekend.strftime('%Y-%m-%d %H:%M')} IST? {is_market_hours_ist(current_ist_time=test_time_weekend)}")


    logger.info(f"Percentage change from 100 to 110: {calculate_percentage_change(100, 110):.2f}%")
    logger.info(f"Percentage change from 100 to 80: {calculate_percentage_change(100, 80):.2f}%")
    
    buy_entry = 100
    profit_pct = 10
    sl_pct = 20 # Risk percentage for SL calculation
    
    tp_buy = calculate_target_price(buy_entry, profit_pct, "BUY")
    sl_buy = calculate_target_price(buy_entry, -sl_pct, "BUY") # SL is negative % for BUY
    logger.info(f"For BUY entry @ {buy_entry}: TP({profit_pct}%) = {tp_buy:.2f}, SL({sl_pct}%) = {sl_buy:.2f}")

    sell_entry = 200
    tp_sell = calculate_target_price(sell_entry, profit_pct, "SELL") # Profit target is lower for SELL
    sl_sell = calculate_target_price(sell_entry, -sl_pct, "SELL") # SL is higher for SELL
    logger.info(f"For SELL entry @ {sell_entry}: TP({profit_pct}%) = {tp_sell:.2f}, SL({sl_pct}%) = {sl_sell:.2f}")

    long_text = "This is a very long string that needs to be truncated for display purposes."
    logger.info(f"Original: {long_text}")
    logger.info(f"Truncated (20): {truncate_string(long_text, 20)}")
    logger.info(f"Truncated (5): {truncate_string(long_text, 5)}") # Suffix might make it longer than max_length if max_length is too small

    # Example of retry decorator
    # @retry_on_exception(max_attempts=2, delay_seconds=1, exceptions_to_catch=(ValueError,))
    # def flaky_function(succeed_on_attempt):
    #     flaky_function.attempts = getattr(flaky_function, 'attempts', 0) + 1
    #     if flaky_function.attempts < succeed_on_attempt:
    #         logger.info(f"flaky_function: Attempt {flaky_function.attempts}, failing...")
    #         raise ValueError("Simulated transient error")
    #     logger.info(f"flaky_function: Attempt {flaky_function.attempts}, succeeding!")
    #     return "Success"
    
    # logger.info("\nTesting retry decorator:")
    # try:
    #     flaky_function.attempts = 0 # Reset for test
    #     result = flaky_function(succeed_on_attempt=2) # Should succeed on 2nd attempt
    #     logger.info(f"Result of flaky_function(2): {result}")
    # except ValueError:
    #     logger.error("flaky_function(2) failed after retries.")
    
    # try:
    #     flaky_function.attempts = 0 # Reset for test
    #     result = flaky_function(succeed_on_attempt=3) # Should fail (max_attempts=2)
    #     logger.info(f"Result of flaky_function(3): {result}") # Won't reach if it raises
    # except ValueError as e:
    #     logger.error(f"flaky_function(3) correctly failed after retries with: {e}")
