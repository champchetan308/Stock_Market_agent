# tests/backtesting/data_loader_bt.py
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class BacktestDataLoader:
    """
    Loads historical market data for backtesting from pre-fetched CSV files.
    Ensures data is correctly formatted with standard column names.
    """
    def __init__(self, data_filepath, 
                 datetime_col_name='Timestamp', # Expected name in CSV after fetching
                 open_col_name='open', 
                 high_col_name='high', 
                 low_col_name='low', 
                 close_col_name='close', 
                 volume_col_name='volume'):
        self.data_filepath = data_filepath
        self.datetime_col = datetime_col_name
        self.open_col = open_col_name
        self.high_col = high_col_name
        self.low_col = low_col_name
        self.close_col = close_col_name
        self.volume_col = volume_col_name
        self.data_df = None # Stores the loaded DataFrame

        if not os.path.exists(self.data_filepath):
            logger.error(f"BacktestDataLoader: Data file not found at path: {self.data_filepath}")
            raise FileNotFoundError(f"Historical data file not found: {self.data_filepath}")

    def load_and_prepare_data(self, start_date_str=None, end_date_str=None):
        """
        Loads data from the CSV file, standardizes columns, and optionally filters by date.
        Args:
            start_date_str (str, optional): "YYYY-MM-DD" format.
            end_date_str (str, optional): "YYYY-MM-DD" format.
        Returns:
            pd.DataFrame: Loaded, processed, and filtered historical data, or None on error.
        """
        try:
            logger.info(f"BacktestDataLoader: Loading historical data from: {self.data_filepath}")
            df = pd.read_csv(self.data_filepath)
            
            # --- Column Name Standardization (ensure they match what historical_data_fetcher saves) ---
            # Assuming historical_data_fetcher already saves with these standard names:
            # 'Timestamp', 'open', 'high', 'low', 'close', 'volume'
            # If not, add a rename map here similar to the one in the old DataLoader.
            
            # Check for required columns
            required_cols = [self.datetime_col, self.open_col, self.high_col, self.low_col, self.close_col, self.volume_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"BacktestDataLoader: Missing required columns in data: {missing_cols}. Expected: {required_cols}")
                return None

            # Convert timestamp column to datetime objects and set as index
            try:
                df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
            except Exception as e:
                logger.error(f"BacktestDataLoader: Error converting timestamp column '{self.datetime_col}' to datetime: {e}. Check data format.", exc_info=True)
                return None
                
            df.set_index(self.datetime_col, inplace=True)
            df.sort_index(inplace=True) # Crucial for time-series processing

            # Filter by date range if provided
            if start_date_str:
                try:
                    start_dt = pd.to_datetime(start_date_str)
                    df = df[df.index >= start_dt]
                except ValueError:
                    logger.error(f"Invalid start_date_str format: {start_date_str}. Use YYYY-MM-DD.")
            if end_date_str:
                try:
                    # Include the entire end_date, so filter up to the start of the next day
                    end_dt = pd.to_datetime(end_date_str) + pd.Timedelta(days=1) 
                    df = df[df.index < end_dt] 
                except ValueError:
                    logger.error(f"Invalid end_date_str format: {end_date_str}. Use YYYY-MM-DD.")


            if df.empty:
                logger.warning(f"BacktestDataLoader: No data found for the filepath '{self.data_filepath}' and specified date range.")
                self.data_df = pd.DataFrame() # Store empty DF
                return self.data_df

            # Ensure numeric types for OHLCV columns and handle potential errors
            for col in [self.open_col, self.high_col, self.low_col, self.close_col, self.volume_col]:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError as ve:
                    logger.error(f"BacktestDataLoader: Error converting column '{col}' to numeric: {ve}. Check data for non-numeric values.")
                    # Optionally, coerce errors: pd.to_numeric(df[col], errors='coerce') and then handle NaNs
                    return None 
            
            # Drop rows where essential price data might be missing or became NaN after coercion
            df.dropna(subset=[self.open_col, self.high_col, self.low_col, self.close_col], inplace=True)

            self.data_df = df
            logger.info(f"BacktestDataLoader: Successfully loaded and prepared {len(self.data_df)} data points "
                        f"from {self.data_df.index.min()} to {self.data_df.index.max()}.")
            return self.data_df

        except FileNotFoundError: # Already checked in init, but good to have here too
            logger.error(f"BacktestDataLoader: Data file not found during load: {self.data_filepath}")
            return None
        except Exception as e:
            logger.error(f"BacktestDataLoader: Unexpected error loading data from {self.data_filepath}: {e}", exc_info=True)
            return None

    def get_data_iterator(self):
        """
        Returns an iterator that yields data row by row as a dictionary.
        Each dict represents a candle: {'timestamp', 'open', 'high', 'low', 'close', 'volume'}.
        """
        if self.data_df is None or self.data_df.empty:
            logger.warning("BacktestDataLoader: Data not loaded or empty. Call load_and_prepare_data() first or check data source.")
            return iter([]) # Empty iterator

        # Iterate over rows, yielding a dictionary for each bar
        # Ensure column names in the yielded dict are the standard ones.
        for timestamp_idx, row_series in self.data_df.iterrows():
            bar_data_dict = {
                'timestamp': timestamp_idx, # Timestamp is the index
                'open': row_series[self.open_col],
                'high': row_series[self.high_col],
                'low': row_series[self.low_col],
                'close': row_series[self.close_col],
                'volume': row_series[self.volume_col]
            }
            # Include any other columns that might be in self.data_df (e.g., from merged sources)
            for col in self.data_df.columns:
                if col not in [self.open_col, self.high_col, self.low_col, self.close_col, self.volume_col]:
                    bar_data_dict[col] = row_series[col]
            yield bar_data_dict

# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Create a dummy CSV for testing (assuming historical_data_fetcher saved it like this)
    dummy_bt_data = {
        'Timestamp': pd.to_datetime(['2023-01-01 09:15:00', '2023-01-01 09:16:00', '2023-01-01 09:17:00', '2023-01-02 09:15:00']),
        'open': [100.1, 101.2, 100.55, 102.0], # Using lowercase to match common output
        'high': [102.3, 101.5, 101.0, 103.5],
        'low': [99.8, 100.0, 100.0, 101.5],
        'close': [101.5, 100.5, 100.88, 102.5],
        'volume': [1000, 1200, 1100, 1500]
    }
    dummy_bt_df = pd.DataFrame(dummy_bt_data)
    dummy_bt_csv_path = "dummy_bt_market_data.csv"
    dummy_bt_df.to_csv(dummy_bt_csv_path, index=False)

    try:
        loader = BacktestDataLoader(dummy_bt_csv_path) # Uses default column names
        data_loaded_df = loader.load_and_prepare_data(start_date_str='2023-01-01', end_date_str='2023-01-01')

        if data_loaded_df is not None and not data_loaded_df.empty:
            logger.info("\nLoaded DataFrame for Backtest (Head):")
            logger.info(data_loaded_df.head())
            
            logger.info("\nIterating through data for backtest:")
            data_iter = loader.get_data_iterator()
            count = 0
            for bar in data_iter:
                if count < 5: logger.info(bar)
                else: break
                count += 1
        else:
            logger.error("Failed to load dummy data for backtest.")
    except FileNotFoundError as fnf:
        logger.error(f"Test failed: {fnf}")
    finally:
        if os.path.exists(dummy_bt_csv_path):
            os.remove(dummy_bt_csv_path)
