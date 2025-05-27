# kite_integration/kite_utils.py
from kiteconnect import KiteConnect, KiteTicker
from kiteconnect.exceptions import KiteException, TokenException, NetworkException, DataException, GeneralException, InputException
import os
import logging
import time
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

# --- Environment Variables for Kite Connect ---
KITE_API_KEY_ENV = os.environ.get("KITE_API_KEY")
KITE_API_SECRET_ENV = os.environ.get("KITE_API_SECRET") # Needed for access token generation
KITE_ACCESS_TOKEN_ENV = os.environ.get("KITE_ACCESS_TOKEN") # This needs to be refreshed daily

class KiteHelper:
    def __init__(self, api_key=None, api_secret=None, access_token=None, max_retries=3, retry_delay_sec=5):
        self.api_key = api_key or KITE_API_KEY_ENV
        self.api_secret = api_secret or KITE_API_SECRET_ENV # Store for potential re-login flows
        self.access_token = access_token or KITE_ACCESS_TOKEN_ENV
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        
        if not self.api_key:
            raise ValueError("Kite API Key is required for KiteHelper.")
        
        self.kite = KiteConnect(api_key=self.api_key)
        
        if self.access_token:
            try:
                self.kite.set_access_token(self.access_token)
                # Verify token validity by making a simple call
                profile = self.kite.profile()
                logger.info(f"KiteHelper initialized successfully. User ID: {profile.get('user_id')}")
            except TokenException as te:
                logger.error(f"Kite TokenException on init: {te}. Access token might be invalid or expired. Manual re-login likely required.", exc_info=True)
                # In a real system, trigger an alert or a re-authentication flow.
                # For now, subsequent calls will likely fail if token is bad.
                self.access_token = None # Invalidate token
            except (KiteException, NetworkException) as ke:
                logger.error(f"KiteException/NetworkException on init: {ke}. Check connectivity or Kite API status.", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error during KiteHelper init with access token: {e}", exc_info=True)
        else:
            logger.warning("KiteHelper initialized without an access token. Most API calls will fail. "
                           "Access token needs to be generated and set via set_access_token() or environment variable.")

    def _make_api_call_with_retry(self, api_call_func, *args, **kwargs):
        """Wrapper to make Kite API calls with retry logic for network/transient errors."""
        for attempt in range(self.max_retries):
            try:
                return api_call_func(*args, **kwargs)
            except TokenException as te: # Non-retryable by this wrapper; token needs refresh
                logger.error(f"Kite TokenException: {te}. Access token invalid/expired. Aborting call.", exc_info=True)
                raise # Re-raise for higher level handling (e.g. re-auth flow)
            except (NetworkException, DataException, GeneralException, InputException) as ke: # Kite specific errors
                # DataException might be retryable if it's a temporary issue, or non-retryable if bad input.
                # InputException is typically non-retryable.
                if isinstance(ke, InputException) or (isinstance(ke, DataException) and "token is not an investor token" in str(ke)): # Example non-retryable data error
                    logger.error(f"Kite non-retryable error: {type(ke).__name__} - {ke}. Aborting call.", exc_info=True)
                    raise
                logger.warning(f"Kite API call failed (Attempt {attempt + 1}/{self.max_retries}): {type(ke).__name__} - {ke}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Max retries reached for Kite API call. Error: {ke}", exc_info=True)
                    raise # Re-raise the last exception
                time.sleep(self.retry_delay_sec * (attempt + 1)) # Incremental delay
            except Exception as e: # Catch-all for other unexpected issues during the call
                logger.error(f"Unexpected error during Kite API call (Attempt {attempt + 1}): {e}", exc_info=True)
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay_sec * (attempt + 1))
        return None # Should not be reached if an exception is always raised on final failure

    def set_access_token(self, access_token):
        """Sets or updates the access token."""
        if not access_token:
            logger.error("Attempted to set an empty access token.")
            return False
        self.access_token = access_token
        self.kite.set_access_token(self.access_token)
        logger.info("Kite access token updated in KiteHelper.")
        try:
            profile = self.kite.profile() # Verify new token
            logger.info(f"Token verification successful. User ID: {profile.get('user_id')}")
            return True
        except TokenException as te:
            logger.error(f"Failed to verify new access token: {te}. Token may be invalid.", exc_info=True)
            self.access_token = None # Invalidate
            return False
        except Exception as e:
            logger.error(f"Error verifying new access token: {e}", exc_info=True)
            return False


    def get_login_url(self):
        """Generates the Kite Connect login URL."""
        if not self.api_key:
            logger.error("Cannot generate login URL without API key.")
            return None
        try:
            return self.kite.login_url()
        except Exception as e:
            logger.error(f"Error generating login URL: {e}", exc_info=True)
            return None

    def generate_session(self, request_token):
        """Generates a session (access token) using a request token."""
        if not self.api_key or not self.api_secret:
            logger.error("API key and secret are required to generate a session.")
            return None
        if not request_token:
            logger.error("Request token is required.")
            return None
        try:
            data = self._make_api_call_with_retry(self.kite.generate_session, request_token, api_secret=self.api_secret)
            if data and "access_token" in data:
                self.set_access_token(data["access_token"])
                logger.info(f"Kite session generated successfully. Access token obtained for user: {data.get('user_id')}")
                return data # Contains access_token, public_token, user_id etc.
            else:
                logger.error(f"Failed to generate session. Response: {data}")
                return None
        except Exception as e: # Catch re-raised exceptions from _make_api_call_with_retry
            logger.error(f"Error during generate_session: {e}", exc_info=True)
            return None

    def get_profile(self):
        if not self.access_token: logger.warning("No access token set for get_profile."); return None
        return self._make_api_call_with_retry(self.kite.profile)

    def get_margins(self):
        if not self.access_token: logger.warning("No access token set for get_margins."); return None
        return self._make_api_call_with_retry(self.kite.margins)
        
    def get_available_capital(self, segment="equity"):
        """Estimates available capital for trading. This is a simplified example."""
        try:
            margins_data = self.get_margins()
            if margins_data and segment in margins_data:
                # Example: For equity, 'net' might represent available cash after considering holdings.
                # For F&O, it's more complex (span, exposure, etc.).
                # This needs to be adapted based on actual margin structure from Kite API.
                available = margins_data[segment].get("available", {}).get("live_balance") # Highly dependent on API response
                if available is None: # Fallback
                    available = margins_data[segment].get("net")

                if available is not None:
                    logger.info(f"Available capital for {segment}: {available}")
                    return float(available)
                else:
                    logger.warning(f"Could not determine available capital from margins data for segment '{segment}'. Data: {margins_data[segment]}")
            else:
                logger.warning(f"Margins data not available or segment '{segment}' not found. Data: {margins_data}")
        except Exception as e:
            logger.error(f"Error fetching or parsing margins for capital: {e}", exc_info=True)
        return 0.0 # Default if unable to fetch


    def place_order(self, trading_symbol, exchange, transaction_type, quantity, 
                    order_type, product, price=None, trigger_price=None, variety=None):
        if not self.access_token: logger.error("No access token. Cannot place order."); return None
        if variety is None: variety = self.kite.VARIETY_REGULAR
        
        logger.info(f"Placing order: {transaction_type} {quantity} {trading_symbol} ({exchange}) "
                    f"Type: {order_type}, Product: {product}, Price: {price}, Trigger: {trigger_price}")
        try:
            order_id = self._make_api_call_with_retry(
                self.kite.place_order,
                tradingsymbol=trading_symbol, exchange=exchange,
                transaction_type=transaction_type, quantity=quantity,
                order_type=order_type, product=product,
                price=price, trigger_price=trigger_price, variety=variety
            )
            logger.info(f"Order placement initiated. Order ID: {order_id}")
            return order_id
        except Exception as e: # Catch re-raised exceptions
            logger.error(f"Order placement failed for {trading_symbol}: {e}", exc_info=True)
            return None

    def get_order_history(self, order_id=None):
        if not self.access_token: logger.warning("No access token set for get_order_history."); return None
        if order_id:
            return self._make_api_call_with_retry(self.kite.order_history, order_id=order_id)
        return self._make_api_call_with_retry(self.kite.orders) # All orders for the day

    # --- Historical Data Fetching (from previous historical_data_fetcher.py, integrated here) ---
    def fetch_historical_data_chunk(self, instrument_token, from_date_dt, to_date_dt, interval):
        """Fetches a single chunk of historical data."""
        if not self.access_token: logger.error("No access token. Cannot fetch historical data."); return None
        logger.debug(f"KiteHelper: Fetching historical chunk {instrument_token} from {from_date_dt} to {to_date_dt} interval {interval}")
        try:
            records = self._make_api_call_with_retry(
                self.kite.historical_data,
                instrument_token=int(instrument_token), # Ensure int
                from_date=from_date_dt, # Expects datetime object
                to_date=to_date_dt,     # Expects datetime object
                interval=interval,
                continuous=False, # Set to True for continuous futures data if needed
                oi=False # Set to True for OI data if needed
            )
            return records
        except TokenException as te: # Specifically handle token issues here too
            logger.error(f"TokenException during historical data fetch: {te}. Token likely invalid.", exc_info=True)
            raise # Re-raise to signal critical auth failure
        except Exception as e:
            logger.error(f"Error fetching historical data chunk for token {instrument_token}: {e}", exc_info=True)
            return None # Return None for this chunk on other errors

    def get_full_historical_data(self, instrument_token, from_date_str, to_date_str, interval="minute"):
        """
        Fetches historical data, handling Kite's date range limitations for minute data by chunking.
        Args:
            instrument_token (int): Instrument token.
            from_date_str (str): "YYYY-MM-DD"
            to_date_str (str): "YYYY-MM-DD"
            interval (str): e.g., "minute", "5minute", "day".
        Returns:
            pd.DataFrame: DataFrame with columns [Timestamp, Open, High, Low, Close, Volume], or empty DataFrame.
        """
        if not self.access_token:
            logger.error("Cannot fetch historical data: Access token not set.")
            return pd.DataFrame()

        try:
            from_date = datetime.strptime(from_date_str, "%Y-%m-%d")
            to_date = datetime.strptime(to_date_str, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format. Use YYYY-MM-DD. Got from='{from_date_str}', to='{to_date_str}'")
            return pd.DataFrame()

        if from_date > to_date:
            logger.error(f"From_date ({from_date_str}) cannot be after to_date ({to_date_str}).")
            return pd.DataFrame()

        logger.info(f"Initiating full historical data fetch for token {instrument_token} ({interval}) "
                    f"from {from_date_str} to {to_date_str}.")

        all_records_list = []
        current_period_start_date = from_date

        # Determine max days per fetch based on interval (Kite limits minute data to ~60-100 days)
        # For day interval, it can be much longer.
        days_per_chunk = 60 if "minute" in interval else 365 # Example chunk sizes

        while current_period_start_date <= to_date:
            period_end_date_ideal = current_period_start_date + timedelta(days=days_per_chunk -1)
            current_period_end_date = min(period_end_date_ideal, to_date)
            
            logger.debug(f"  Fetching data for period: {current_period_start_date.strftime('%Y-%m-%d')} "
                         f"to {current_period_end_date.strftime('%Y-%m-%d')}")
            
            try:
                # Kite API needs time for 'to_date' for minute data to include full day.
                # Let's use end of day for to_date for minute intervals.
                api_to_date = current_period_end_date
                if "minute" in interval:
                     api_to_date = datetime.combine(current_period_end_date, datetime.max.time())


                chunk_records = self.fetch_historical_data_chunk(
                    instrument_token, 
                    current_period_start_date, # This is datetime object
                    api_to_date,               # This is datetime object
                    interval
                )
                if chunk_records:
                    all_records_list.extend(chunk_records)
                    logger.info(f"    Fetched {len(chunk_records)} records for this period.")
                else:
                    logger.info(f"    No records for period ending {current_period_end_date.strftime('%Y-%m-%d')}.")
                
                # API rate limiting
                time.sleep(0.3) # Adjust as needed based on Kite's limits (e.g., 3 req/sec)

            except TokenException: # Propagated from fetch_historical_data_chunk
                logger.error("TokenException encountered during historical data fetch. Aborting further fetches.")
                return pd.DataFrame(all_records_list) # Return what's fetched so far
            except Exception as e:
                logger.error(f"  Error in fetching/processing chunk ending {current_period_end_date.strftime('%Y-%m-%d')}: {e}", exc_info=True)
                # Decide whether to break or continue on other errors. For now, continue.

            current_period_start_date = current_period_end_date + timedelta(days=1)
            if current_period_start_date > to_date and not all_records_list: # Ensure loop terminates if to_date was small
                break


        if not all_records_list:
            logger.warning(f"No historical data retrieved for token {instrument_token} in the specified range.")
            return pd.DataFrame()

        # Process into DataFrame
        hist_df = pd.DataFrame(all_records_list)
        if 'date' in hist_df.columns: # Kite returns 'date' as timestamp column
            hist_df.rename(columns={'date': 'Timestamp'}, inplace=True)
        else: # Should not happen if API is consistent
            logger.error("Timestamp column ('date') not found in historical data from Kite.")
            return pd.DataFrame()
            
        hist_df['Timestamp'] = pd.to_datetime(hist_df['Timestamp'])
        # Filter again just to be sure dates are within requested range (API might give boundary data)
        hist_df = hist_df[(hist_df['Timestamp'] >= from_date) & (hist_df['Timestamp'] < (to_date + timedelta(days=1)))] # Up to end of to_date
        
        hist_df.sort_values(by='Timestamp', inplace=True)
        hist_df.drop_duplicates(subset=['Timestamp'], keep='first', inplace=True)
        hist_df.reset_index(drop=True, inplace=True)
        
        # Ensure required columns (Open, High, Low, Close, Volume)
        expected_cols = ['Timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            if col not in hist_df.columns and col != 'Timestamp': # Timestamp already handled
                 logger.warning(f"Column '{col}' missing in historical data. Filling with 0 or NaN if critical.")
                 hist_df[col] = 0 # Or pd.NA
        
        logger.info(f"Total {len(hist_df)} unique historical records processed for token {instrument_token}.")
        return hist_df[['Timestamp', 'open', 'high', 'low', 'close', 'volume']]


    def get_instruments(self, exchange=None):
        """Fetches list of instruments. Can be large, cache if possible."""
        if not self.access_token: logger.warning("No access token for get_instruments."); return None
        return self._make_api_call_with_retry(self.kite.instruments, exchange=exchange)

    def get_ltp(self, instruments_list):
        """Get LTP for a list of instruments. instruments_list: e.g., ["NSE:INFY", "NFO:NIFTY23JULPUT"] or [token1, token2]"""
        if not self.access_token: logger.warning("No access token for get_ltp."); return None
        if not instruments_list: return {}
        return self._make_api_call_with_retry(self.kite.ltp, instruments=instruments_list)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("--- Testing KiteHelper ---")
    
    if not KITE_API_KEY_ENV or not KITE_ACCESS_TOKEN_ENV: # Add KITE_API_SECRET_ENV for session generation test
        logger.error("KITE_API_KEY and KITE_ACCESS_TOKEN (and KITE_API_SECRET for session test) "
                       "must be set as environment variables for this test.")
    else:
        try:
            helper = KiteHelper()
            
            profile_data = helper.get_profile()
            if profile_data:
                logger.info(f"Profile: {profile_data.get('user_name')}, Email: {profile_data.get('email')}")
            else:
                logger.error("Failed to get profile. Access token might be invalid.")

            # --- Test Historical Data Fetch (use with caution, respects API limits) ---
            # nifty_token_example = 256265 # NIFTY 50
            # from_date_example = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            # to_date_example = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            # logger.info(f"\nFetching NIFTY 50 '5minute' data from {from_date_example} to {to_date_example}")
            # nifty_hist_df = helper.get_full_historical_data(
            #     instrument_token=nifty_token_example,
            #     from_date_str=from_date_example,
            #     to_date_str=to_date_example,
            #     interval="5minute"
            # )
            # if not nifty_hist_df.empty:
            #     logger.info(f"Fetched NIFTY 50 data. Shape: {nifty_hist_df.shape}")
            #     logger.info(f"Head:\n{nifty_hist_df.head()}")
            #     logger.info(f"Tail:\n{nifty_hist_df.tail()}")
            #     # nifty_hist_df.to_csv("nifty_5min_test_data.csv", index=False) # Optional: save
            # else:
            #     logger.warning("No NIFTY 50 historical data returned for the test period.")

        except ValueError as ve: # From KiteHelper init if API key missing
            logger.error(f"Initialization error: {ve}")
        except TokenException as te_main:
             logger.error(f"TokenException in main test: {te_main}. Access token is likely invalid or expired.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in KiteHelper test: {e}", exc_info=True)
