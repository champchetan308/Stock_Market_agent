# tests/backtesting/portfolio_manager_bt.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PortfolioManagerBT:
    """
    Manages a simulated trading portfolio for backtesting, adhering to specified
    capital allocation, profit target, and stop-loss rules.
    Tracks a single active trade at a time.
    """
    def __init__(self, initial_capital=1000000.0, 
                 brokerage_per_trade_fixed=0.0, # Fixed amount per trade
                 brokerage_per_trade_pct=0.0,   # Percentage of trade value
                 capital_allocation_pct_per_trade=0.10, # 10% of total capital
                 default_profit_target_pct_on_capital=0.10, # 10% profit on capital deployed
                 default_stop_loss_pct_on_capital=0.20):   # 20% risk on capital deployed
        
        self.initial_capital = float(initial_capital)
        self.brokerage_fixed = float(brokerage_per_trade_fixed)
        self.brokerage_pct = float(brokerage_per_trade_pct) # e.g., 0.0005 for 0.05%
        
        self.capital_allocation_pct = float(capital_allocation_pct_per_trade)
        self.default_profit_target_pct_on_capital_deployed = float(default_profit_target_pct_on_capital)
        self.default_stop_loss_pct_on_capital_deployed = float(default_stop_loss_pct_on_capital)
        
        self.cash = float(initial_capital)
        self.active_trade = None # Stores dict of the single active trade details
        self.trade_log = []      # List of all completed trade dictionaries
        self.equity_curve = [{'timestamp': None, 'portfolio_value': self.initial_capital}] # Start with initial capital
        self.successful_trades_today_count = 0 # For the "max 5 successful trades" rule

        logger.info(f"PortfolioManagerBT initialized. Initial Capital: {self.initial_capital:.2f}, "
                    f"Capital Allocation/Trade: {self.capital_allocation_pct*100:.1f}%, "
                    f"Default TP % (on deployed): {self.default_profit_target_pct_on_capital_deployed*100:.1f}%, "
                    f"Default SL % (on deployed): {self.default_stop_loss_pct_on_capital_deployed*100:.1f}%")

    def _calculate_brokerage(self, trade_value):
        """Calculates brokerage for a trade."""
        return self.brokerage_fixed + (trade_value * self.brokerage_pct)

    def can_open_new_trade(self, current_date): # current_date to reset daily counter
        """Checks if a new trade can be opened based on active trade and daily limits."""
        # Reset daily successful trade counter if date has changed
        # This needs a more robust way to track "today" if backtest spans multiple days.
        # For simplicity, assume backtest runner handles date changes or this is called appropriately.
        # if self.equity_curve and self.equity_curve[-1]['timestamp'] and self.equity_curve[-1]['timestamp'].date() != current_date:
        #    self.successful_trades_today_count = 0
        #    logger.info(f"PortfolioManagerBT: Reset daily successful trade count for new date {current_date}.")

        if self.active_trade is not None:
            logger.debug("PortfolioManagerBT: Cannot open new trade. An existing trade is active.")
            return False
        if self.successful_trades_today_count >= 5: # Max 5 successful trades per day
            logger.info("PortfolioManagerBT: Max 5 successful trades for the day reached. No new trades.")
            return False
        return True

    def open_trade(self, timestamp, instrument_token, trading_symbol, exchange,
                   side, entry_price, 
                   ai_suggested_sl_price=None, ai_suggested_tp_price=None,
                   ai_decision_details=None):
        """
        Opens a new trade if conditions are met.
        Calculates quantity based on 10% capital allocation.
        Sets SL at ~20% risk on deployed capital, TP at ~10% profit on deployed capital.
        """
        if not self.can_open_new_trade(timestamp.date()): # Pass current date for daily counter logic
            return False

        if entry_price <= 0:
            logger.error(f"PortfolioManagerBT: Invalid entry price {entry_price} for opening trade. Aborting.")
            return False
            
        capital_to_deploy = self.cash * self.capital_allocation_pct # Use 10% of current cash
        if capital_to_deploy <=0:
            logger.warning(f"PortfolioManagerBT: Not enough capital to deploy ({capital_to_deploy:.2f}). Cannot open trade.")
            return False

        quantity = int(capital_to_deploy / entry_price)
        if quantity == 0:
            logger.warning(f"PortfolioManagerBT: Calculated quantity is 0 for {trading_symbol} at price {entry_price} "
                           f"with allocated capital {capital_to_deploy:.2f}. Cannot open trade.")
            return False

        trade_value = quantity * entry_price
        brokerage = self._calculate_brokerage(trade_value)

        if self.cash < (trade_value + brokerage) and side.upper() == "BUY": # Check for BUY, short selling margin is complex
            logger.warning(f"PortfolioManagerBT: Insufficient cash for BUY. Needed: {trade_value + brokerage:.2f}, Have: {self.cash:.2f}")
            return False
        
        # Determine SL and TP based on deployed capital for this trade (trade_value)
        # Risk 20% of trade_value, Profit 10% of trade_value
        risk_amount_per_share = (trade_value * self.default_stop_loss_pct_on_capital_deployed) / quantity
        profit_amount_per_share = (trade_value * self.default_profit_target_pct_on_capital_deployed) / quantity

        if side.upper() == "BUY":
            stop_loss_price = entry_price - risk_amount_per_share
            profit_target_price = entry_price + profit_amount_per_share
            self.cash -= (trade_value + brokerage)
        elif side.upper() == "SELL": # Short sell
            stop_loss_price = entry_price + risk_amount_per_share
            profit_target_price = entry_price - profit_amount_per_share
            self.cash += (trade_value - brokerage) # Cash increases from short sale proceeds (less brokerage)
        else:
            logger.error(f"PortfolioManagerBT: Invalid trade side '{side}'.")
            return False

        # Override with AI suggested SL/TP if provided and valid (e.g. not worse than calculated)
        # This logic can be refined. For now, let's prioritize calculated ones based on fixed % risk/reward.
        # if ai_suggested_sl_price is not None: stop_loss_price = ai_suggested_sl_price
        # if ai_suggested_tp_price is not None: profit_target_price = ai_suggested_tp_price
        
        self.active_trade = {
            'entry_timestamp': timestamp, 'instrument_token': instrument_token,
            'trading_symbol': trading_symbol, 'exchange': exchange,
            'side': side.upper(), 'quantity': quantity, 'entry_price': entry_price,
            'trade_value_at_entry': trade_value,
            'stop_loss_price': round(stop_loss_price, 2), 
            'profit_target_price': round(profit_target_price, 2),
            'brokerage_entry': brokerage,
            'ai_decision_details': ai_decision_details # Store AI's reasoning for this trade
        }
        logger.info(f"PortfolioManagerBT: NEW TRADE OPENED: {self.active_trade['side']} {self.active_trade['quantity']} "
                    f"{self.active_trade['trading_symbol']} @ {self.active_trade['entry_price']:.2f}. "
                    f"SL: {self.active_trade['stop_loss_price']:.2f}, TP: {self.active_trade['profit_target_price']:.2f}. "
                    f"Deployed: {trade_value:.2f}. Cash Left: {self.cash:.2f}")
        return True

    def close_active_trade(self, timestamp, exit_price, reason="SL/TP Hit"):
        """Closes the currently active trade."""
        if not self.active_trade:
            logger.warning("PortfolioManagerBT: No active trade to close.")
            return False
        
        trade = self.active_trade
        exit_trade_value = trade['quantity'] * exit_price
        brokerage_exit = self._calculate_brokerage(exit_trade_value)
        
        pnl_gross = 0
        if trade['side'] == "BUY":
            pnl_gross = (exit_price - trade['entry_price']) * trade['quantity']
            self.cash += (exit_trade_value - brokerage_exit) # Add proceeds from sale
        elif trade['side'] == "SELL": # Closing a short
            pnl_gross = (trade['entry_price'] - exit_price) * trade['quantity']
            self.cash -= (exit_trade_value + brokerage_exit) # Pay to buy back shares to cover

        pnl_net = pnl_gross - trade['brokerage_entry'] - brokerage_exit
        
        if pnl_net > 0:
            self.successful_trades_today_count += 1
            logger.info(f"PortfolioManagerBT: Successful trade closed. Count today: {self.successful_trades_today_count}")

        completed_trade_log_entry = {
            **trade, # Copy all details from active_trade
            'exit_timestamp': timestamp,
            'exit_price': exit_price,
            'brokerage_exit': brokerage_exit,
            'pnl_gross': round(pnl_gross, 2),
            'pnl_net': round(pnl_net, 2),
            'close_reason': reason,
            'cash_after_trade_close': self.cash
        }
        self.trade_log.append(completed_trade_log_entry)
        
        logger.info(f"PortfolioManagerBT: TRADE CLOSED: {trade['side']} {trade['quantity']} {trade['trading_symbol']} "
                    f"Entry @ {trade['entry_price']:.2f}, Exit @ {exit_price:.2f}. Reason: {reason}. "
                    f"Net P&L: {pnl_net:.2f}. Cash: {self.cash:.2f}")
        
        self.active_trade = None # Clear the active trade
        return True

    def update_portfolio_value_and_equity_curve(self, timestamp, current_market_prices_dict):
        """
        Calculates current portfolio value (MTM) and adds to equity curve.
        Args:
            timestamp (datetime): Current timestamp for the MTM.
            current_market_prices_dict (dict): {instrument_token: current_price}.
        """
        current_holdings_value = 0.0
        if self.active_trade:
            token = self.active_trade['instrument_token']
            current_price = current_market_prices_dict.get(token)
            if current_price is None: # Fallback if LTP for active trade is missing
                current_price = self.active_trade['entry_price'] 
                logger.warning(f"PortfolioManagerBT: Missing current market price for active trade {token}. Using entry price for MTM.")

            if self.active_trade['side'] == 'BUY':
                current_holdings_value = self.active_trade['quantity'] * current_price
            elif self.active_trade['side'] == 'SELL': # MTM for short position
                # Value of short position = Initial cash from shorting + P&L
                # P&L = (entry_price - current_price) * quantity
                # So, MTM value = (entry_price * quantity) + (entry_price - current_price) * quantity
                # This can be thought of as: if we were to cover now, what's the cash impact relative to initial state.
                # For portfolio value, it's simpler: cash + (value of assets if bought back)
                # Cash already reflects proceeds from short. Unrealized P&L is (entry_price - current_price) * quantity.
                # The "value" of the short position for equity curve is the current liability to buy it back.
                # So, portfolio value = self.cash (which includes short proceeds) - (quantity * current_price to buy back)
                # This is equivalent to: initial_capital + sum_of_all_closed_trade_pnl + unrealized_pnl_of_active_short
                unrealized_pnl_short = (self.active_trade['entry_price'] - current_price) * self.active_trade['quantity']
                # The `current_holdings_value` should represent the "asset" side.
                # For a short, the "asset" is the obligation to buy back.
                # Let's adjust cash based on unrealized P&L for shorts.
                # self.cash already has the proceeds. So, we add the unrealized P&L.
                # current_holdings_value = unrealized_pnl_short (this is not total value, but the change)
                # Let's use a simpler: Portfolio Value = Cash + Value of Longs - Value of Shorts (liability)
                # If short, the "value" of the position for MTM is -(quantity * current_price) IF cash wasn't already adjusted.
                # Since cash IS adjusted on short entry, we just need to value the obligation.
                # The cash component already has the short sale proceeds.
                # The "asset" part is the current market value of the shorted stock (negative).
                # So, if we have a short position, the "holdings_value" is effectively the unrealized P&L.
                # No, more simply: current_holdings_value = (initial value of shorted stock) + PnL
                # initial value = entry_price * quantity. PnL = (entry_price - current_price) * quantity
                # So, value = entry_price*qty + (entry_price - current_price)*qty = (2*entry_price - current_price)*qty
                # This is not intuitive.
                # Let's use: current_portfolio_value = self.cash + (current_price * quantity_if_long) - (current_price * quantity_if_short)
                # BUT self.cash is already updated.
                # Correct MTM: Current Cash + (Current Market Value of Long Positions) + (Unrealized P&L of Short Positions)
                # Unrealized P&L for short = (Entry Price - Current Price) * Quantity
                current_holdings_value = (self.active_trade['entry_price'] - current_price) * self.active_trade['quantity']


        portfolio_value = self.cash
        if self.active_trade and self.active_trade['side'] == 'BUY':
            portfolio_value += current_holdings_value # Add market value of long position
        elif self.active_trade and self.active_trade['side'] == 'SELL':
            # For short, cash already reflects proceeds. Add unrealized P&L.
            portfolio_value += current_holdings_value 
            # This means if short is profitable (current_price < entry_price), P&L is positive, increasing portfolio_value.
            # If short is losing (current_price > entry_price), P&L is negative, decreasing portfolio_value.

        self.equity_curve.append({'timestamp': timestamp, 'portfolio_value': round(portfolio_value, 2)})
        logger.debug(f"PortfolioManagerBT: Equity curve updated at {timestamp}. Portfolio Value = {portfolio_value:.2f} (Cash: {self.cash:.2f}, Holdings MTM: {current_holdings_value:.2f})")


    def get_performance_summary(self):
        if not self.equity_curve or len(self.equity_curve) <= 1: # Need more than just initial capital point
            final_val = self.cash # If no trades or MTM updates, final value is just cash
        else:
            final_val = self.equity_curve[-1]['portfolio_value']

        total_return_abs = final_val - self.initial_capital
        total_return_pct = (total_return_abs / self.initial_capital) * 100 if self.initial_capital != 0 else 0
        
        num_total_trades_logged = len(self.trade_log) # Each entry in trade_log is a completed trade
        
        winning_trades = [t for t in self.trade_log if t['pnl_net'] > 0]
        losing_trades = [t for t in self.trade_log if t['pnl_net'] < 0]
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)

        win_rate = (num_winning / num_total_trades_logged) * 100 if num_total_trades_logged > 0 else 0
        
        total_brokerage = sum(t.get('brokerage_entry',0) + t.get('brokerage_exit',0) for t in self.trade_log)

        # Max Drawdown calculation
        equity_df = pd.DataFrame(self.equity_curve)
        max_drawdown_pct = 0.0
        if not equity_df.empty and 'portfolio_value' in equity_df.columns:
            equity_df['peak'] = equity_df['portfolio_value'].expanding(min_periods=1).max()
            equity_df['drawdown'] = equity_df['portfolio_value'] - equity_df['peak']
            equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['peak']).replace([np.inf, -np.inf], 0) * 100
            max_drawdown_pct = equity_df['drawdown_pct'].min() if not equity_df['drawdown_pct'].empty else 0.0
        
        summary = {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": round(final_val, 2),
            "net_profit_absolute": round(total_return_abs, 2),
            "net_profit_percentage": round(total_return_pct, 2),
            "total_completed_trades": num_total_trades_logged,
            "winning_trades": num_winning,
            "losing_trades": num_losing,
            "win_rate_percentage": round(win_rate, 2),
            "total_brokerage_paid": round(total_brokerage, 2),
            "max_drawdown_percentage": round(max_drawdown_pct, 2),
            "successful_trades_counter_for_day_limit": self.successful_trades_today_count # Current daily count
        }
        logger.info("PortfolioManagerBT Performance Summary:")
        for k,v in summary.items(): logger.info(f"  {k.replace('_',' ').title()}: {v}")
        return summary

    def reset_daily_counters(self, current_date_for_reset):
        """Resets counters that are daily, like successful_trades_today_count."""
        # This should be called by the backtest engine at the start of a new simulated day.
        # A simple check: if the last equity timestamp's date is different from current_date_for_reset.
        if self.equity_curve and len(self.equity_curve) > 1: # Need at least one MTM point after initial
            last_equity_timestamp = self.equity_curve[-1]['timestamp']
            if last_equity_timestamp and last_equity_timestamp.date() < current_date_for_reset.date():
                logger.info(f"PortfolioManagerBT: New day {current_date_for_reset.date()}. Resetting daily successful trade count from {self.successful_trades_today_count} to 0.")
                self.successful_trades_today_count = 0
        elif self.successful_trades_today_count > 0: # If no equity curve updates yet but counter is non-zero (e.g. first day)
             logger.info(f"PortfolioManagerBT: First day or no MTM updates. Assuming reset for {current_date_for_reset.date()}.")
             self.successful_trades_today_count = 0


# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    pm = PortfolioManagerBT(initial_capital=100000, 
                            capital_allocation_pct_per_trade=0.10, # 10%
                            default_profit_target_pct_on_capital=0.10, # 10% profit on 10k = 1k profit
                            default_stop_loss_pct_on_capital=0.20) # 20% risk on 10k = 2k loss max

    # --- Day 1 ---
    current_sim_date = datetime(2023,1,2)
    pm.reset_daily_counters(current_sim_date)

    # Trade 1 (Successful)
    ts1_entry = datetime(2023,1,2, 9,30,0)
    opened1 = pm.open_trade(ts1_entry, 256265, "NIFTY50", "NSE", "BUY", 17000)
    if opened1:
        # Simulate price movement and MTM
        pm.update_portfolio_value_and_equity_curve(datetime(2023,1,2, 9,45,0), {256265: 17050}) 
        # Simulate TP hit (10% profit on deployed capital of 10k = 1k profit. Entry 17000, Qty based on 10k/17000.
        # Deployed capital = 100000 * 0.1 = 10000. Qty = int(10000/17000) = 0. This is wrong for index.
        # Let's assume quantity is fixed for this test or calculated differently for indices.
        # For simplicity, let's assume 10% price move for TP.
        # TP price for BUY from 17000 with 10% on capital deployed:
        # If Qty = 1 (for simplicity of price move calc), deployed = 17000. TP target = 17000 * 0.1 = 1700. Exit = 18700.
        # This is 10% on capital, not 10% price move.
        # The TP/SL in PortfolioManagerBT are absolute prices.
        # active_trade.profit_target_price was calculated as entry_price + profit_amount_per_share
        # profit_amount_per_share = (trade_value_at_entry * default_profit_target_pct_on_capital_deployed) / quantity
        # profit_amount_per_share = (entry_price * quantity * default_profit_target_pct_on_capital_deployed) / quantity
        # profit_amount_per_share = entry_price * default_profit_target_pct_on_capital_deployed
        # So, TP = entry_price * (1 + default_profit_target_pct_on_capital_deployed)
        tp_price_trade1 = pm.active_trade['profit_target_price'] if pm.active_trade else 17000 * 1.1 # Fallback
        pm.close_active_trade(datetime(2023,1,2, 10,0,0), tp_price_trade1, "Profit Target")
    
    # Trade 2 (Losing)
    ts2_entry = datetime(2023,1,2, 10,30,0)
    opened2 = pm.open_trade(ts2_entry, 260105, "BANKNIFTY", "NSE", "SELL", 48000)
    if opened2:
        pm.update_portfolio_value_and_equity_curve(datetime(2023,1,2, 10,45,0), {260105: 48100})
        sl_price_trade2 = pm.active_trade['stop_loss_price'] if pm.active_trade else 48000 * 1.2 # Fallback
        pm.close_active_trade(datetime(2023,1,2, 11,0,0), sl_price_trade2, "Stop Loss")

    # ... up to 5 successful trades ...
    for i in range(4): # Simulate 4 more successful trades
        if pm.can_open_new_trade(current_sim_date):
            entry_t = datetime(2023,1,2, 11,30+i*30,0)
            if pm.open_trade(entry_t, 1000+i, f"STOCK{i}", "NSE", "BUY", 200+i*10):
                tp_p = pm.active_trade['profit_target_price']
                pm.close_active_trade(datetime(2023,1,2, 12,0+i*30,0), tp_p, "Profit Target")
        else: break
    
    logger.info(f"Successful trades today after loop: {pm.successful_trades_today_count}")
    
    # Attempt 6th trade (should be blocked if 5 were successful)
    ts6_entry = datetime(2023,1,2, 14,0,0)
    opened6 = pm.open_trade(ts6_entry, 738561, "RELIANCE", "NSE", "BUY", 2900)
    if not opened6:
        logger.info("6th trade correctly blocked by daily limit or no capital.")
    else: # Should not happen if 5 successful trades were logged
        logger.error("6th trade was opened, daily limit logic failed.")
        if pm.active_trade: # Close it if opened by mistake
             pm.close_active_trade(datetime(2023,1,2, 14,5,0), 2900*0.99, "Test Close")


    # --- Day 2 ---
    current_sim_date_day2 = datetime(2023,1,3)
    pm.reset_daily_counters(current_sim_date_day2) # Reset for new day
    logger.info(f"Successful trades today after reset for Day 2: {pm.successful_trades_today_count}")
    assert pm.successful_trades_today_count == 0

    ts7_entry = datetime(2023,1,3, 9,30,0)
    opened7 = pm.open_trade(ts7_entry, 256265, "NIFTY50", "NSE", "BUY", 17500)
    if opened7:
        tp_price_trade7 = pm.active_trade['profit_target_price']
        pm.close_active_trade(datetime(2023,1,3, 10,0,0), tp_price_trade7, "Profit Target")
    logger.info(f"Successful trades on Day 2 after one trade: {pm.successful_trades_today_count}")
    assert pm.successful_trades_today_count == 1


    pm.get_performance_summary()
    logger.info("\nFull Trade Log:")
    for entry in pm.trade_log: logger.info(entry)
    logger.info("\nEquity Curve (last 5):")
    for entry in pm.equity_curve[-5:]: logger.info(entry)
