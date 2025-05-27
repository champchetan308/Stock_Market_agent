# tests/backtesting/backtest_analysis.py
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

class BacktestTradeAnalyzer:
    def __init__(self, trade_log_df, portfolio_summary):
        """
        Analyzes a log of trades from a backtest.
        Args:
            trade_log_df (pd.DataFrame): DataFrame of trades from PortfolioManager.
                                         Expected columns: 'timestamp', 'instrument_token', 
                                         'trading_symbol', 'side', 'quantity', 'price', 
                                         'brokerage', 'pnl' (if completed trade).
                                         For AI trades, it might also have 'ai_decision_details',
                                         'contributing_sub_agent_signals'.
            portfolio_summary (dict): Summary dictionary from PortfolioManager.
        """
        self.trade_log_df = trade_log_df
        self.portfolio_summary = portfolio_summary
        self.paired_trades_df = self._pair_trades()

    def _pair_trades(self):
        """
        Attempts to pair BUY and SELL trades for the same instrument to calculate P&L per round trip.
        This is a simplified pairing logic. Assumes one position open at a time per instrument.
        More complex logic needed for partial fills, scaling in/out.
        """
        if self.trade_log_df.empty:
            return pd.DataFrame()

        paired_trades = []
        open_position = {} # Key: instrument_token, Value: entry_trade_details

        # Sort trades by instrument and then by time to process sequentially for each instrument
        sorted_trades = self.trade_log_df.sort_values(by=['instrument_token', 'timestamp'])

        for idx, current_trade in sorted_trades.iterrows():
            token = current_trade['instrument_token']
            
            if token not in open_position: # No open position for this instrument, this trade is an entry
                if current_trade['side'] in ['BUY', 'SELL']: # Valid entry sides
                    open_position[token] = current_trade.copy() # Store copy
            else: # There is an open position for this instrument
                entry_trade = open_position[token]
                # Check if current trade closes or reduces the open position
                if (entry_trade['side'] == 'BUY' and current_trade['side'] == 'SELL') or \
                   (entry_trade['side'] == 'SELL' and current_trade['side'] == 'BUY'):
                    
                    # Simple pairing: assume full close of position for now
                    # A more robust system would handle partial closes.
                    qty_matched = min(entry_trade['quantity'], current_trade['quantity'])
                    
                    if qty_matched > 0:
                        pnl = 0
                        if entry_trade['side'] == 'BUY':
                            pnl = (current_trade['price'] - entry_trade['price']) * qty_matched
                        else: # Entry was SELL (short)
                            pnl = (entry_trade['price'] - current_trade['price']) * qty_matched
                        
                        # Subtract brokerage for both entry and exit
                        total_brokerage_for_round_trip = entry_trade.get('brokerage',0) + current_trade.get('brokerage',0)
                        net_pnl = pnl - total_brokerage_for_round_trip

                        paired_trades.append({
                            'instrument_token': token,
                            'trading_symbol': entry_trade['trading_symbol'],
                            'entry_timestamp': entry_trade['timestamp'],
                            'entry_side': entry_trade['side'],
                            'entry_price': entry_trade['price'],
                            'exit_timestamp': current_trade['timestamp'],
                            'exit_price': current_trade['price'],
                            'quantity': qty_matched,
                            'gross_pnl': pnl,
                            'brokerage': total_brokerage_for_round_trip,
                            'net_pnl': net_pnl,
                            'entry_ai_decision': entry_trade.get('ai_decision_details'), # If available in log
                            'entry_sub_signals': entry_trade.get('contributing_sub_agent_signals'), # If available
                            'exit_reason': current_trade.get('exit_reason', 'Paired Close') # If available
                        })
                        
                        # For this simple model, assume full close and clear open position
                        del open_position[token] 
                        # If partial closes were handled, you'd reduce open_position[token]['quantity']
                else: # Same side trade, e.g. averaging down/up (not handled by this simple pairer)
                    logger.warning(f"Trade pairing: Encountered same-side trade for open position on {token} at {current_trade['timestamp']}. Simple pairing logic might misinterpret.")
                    # For now, replace open position if it's an add-on, or handle as per strategy.
                    # This simple pairer assumes entry then exit.
                    open_position[token] = current_trade.copy()


        return pd.DataFrame(paired_trades)

    def calculate_win_loss_stats(self):
        if self.paired_trades_df.empty:
            return {"message": "No paired trades to analyze for win/loss."}

        total_paired_trades = len(self.paired_trades_df)
        winning_trades_df = self.paired_trades_df[self.paired_trades_df['net_pnl'] > 0]
        losing_trades_df = self.paired_trades_df[self.paired_trades_df['net_pnl'] < 0]
        breakeven_trades_df = self.paired_trades_df[self.paired_trades_df['net_pnl'] == 0]

        num_winning_trades = len(winning_trades_df)
        num_losing_trades = len(losing_trades_df)
        
        win_rate = (num_winning_trades / total_paired_trades) * 100 if total_paired_trades > 0 else 0
        loss_rate = (num_losing_trades / total_paired_trades) * 100 if total_paired_trades > 0 else 0

        avg_profit_per_winning_trade = winning_trades_df['net_pnl'].mean() if num_winning_trades > 0 else 0
        avg_loss_per_losing_trade = losing_trades_df['net_pnl'].mean() if num_losing_trades > 0 else 0 # Will be negative

        profit_factor = abs(winning_trades_df['net_pnl'].sum() / losing_trades_df['net_pnl'].sum()) if num_losing_trades > 0 and losing_trades_df['net_pnl'].sum() !=0 else float('inf') if winning_trades_df['net_pnl'].sum() > 0 else 0

        stats = {
            "total_paired_trades": total_paired_trades,
            "num_winning_trades": num_winning_trades,
            "num_losing_trades": num_losing_trades,
            "num_breakeven_trades": len(breakeven_trades_df),
            "win_rate_percentage": round(win_rate, 2),
            "loss_rate_percentage": round(loss_rate, 2),
            "average_profit_per_winning_trade": round(avg_profit_per_winning_trade, 2),
            "average_loss_per_losing_trade": round(avg_loss_per_losing_trade, 2), # Negative
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "Infinite (no losses)"
        }
        logger.info("Win/Loss Statistics (Paired Trades):")
        for k,v in stats.items(): logger.info(f"  {k.replace('_',' ').title()}: {v}")
        return stats, winning_trades_df, losing_trades_df

    def analyze_failed_trades_report(self, losing_trades_df, accuracy_threshold=50.0):
        """
        Generates a conceptual report for failed trades if overall win rate is below threshold.
        Args:
            losing_trades_df (pd.DataFrame): DataFrame of losing trades.
            accuracy_threshold (float): The win rate percentage below which this report is emphasized.
        """
        overall_win_rate = self.portfolio_summary.get("win_rate_percentage", 
                                 self.calculate_win_loss_stats()[0].get("win_rate_percentage", 100) # Recalc if not in summary
                                )

        if overall_win_rate >= accuracy_threshold:
            logger.info(f"Overall win rate ({overall_win_rate:.2f}%) is above threshold ({accuracy_threshold:.2f}%). Detailed failure analysis not critical.")
            return None

        logger.warning(f"Overall win rate ({overall_win_rate:.2f}%) is BELOW threshold ({accuracy_threshold:.2f}%). Analyzing losing trades...")
        
        if losing_trades_df.empty:
            logger.info("No losing trades to analyze for failure report.")
            return None

        failure_analysis_report = []
        logger.info("\n--- FAILED TRADES ANALYSIS REPORT (Conceptual) ---")
        for idx, trade in losing_trades_df.iterrows():
            report_item = {
                "trade_index": idx,
                "symbol": trade.get('trading_symbol'),
                "entry_side": trade.get('entry_side'),
                "entry_price": trade.get('entry_price'),
                "exit_price": trade.get('exit_price'),
                "net_pnl": trade.get('net_pnl'),
                "entry_timestamp": trade.get('entry_timestamp').strftime('%Y-%m-%d %H:%M:%S') if pd.notna(trade.get('entry_timestamp')) else 'N/A',
                "exit_timestamp": trade.get('exit_timestamp').strftime('%Y-%m-%d %H:%M:%S') if pd.notna(trade.get('exit_timestamp')) else 'N/A',
                "exit_reason": trade.get('exit_reason', 'N/A')
            }
            
            # --- Deeper AI Analysis (Conceptual) ---
            # If AI decision details and sub-agent signals were logged with the trade:
            ai_decision_at_entry = trade.get('entry_ai_decision') # This would be the Main AI's decision JSON
            sub_signals_at_entry = trade.get('entry_sub_signals') # Dict of sub-agent signals
            
            ai_reasoning_at_entry = "N/A"
            if isinstance(ai_decision_at_entry, dict):
                ai_reasoning_at_entry = ai_decision_at_entry.get('reasoning', ai_decision_at_entry.get('primary_reasoning', "No detailed AI reasoning logged."))
            
            report_item["ai_entry_reasoning"] = ai_reasoning_at_entry
            
            # What could be analyzed further (requires historical context around the trade):
            # 1. Market conditions at entry vs. AI's assessment.
            # 2. Conflicting sub-agent signals that might have been overlooked or underweighted by Main AI.
            # 3. How did price move immediately after entry vs. AI's expectation?
            # 4. Was the stop-loss appropriate for volatility at the time?
            # 5. If Main AI used Gemini, what was the exact prompt and full Gemini response for this trade?
            # This level of detail would need access to the full context data around the trade time.
            
            logger.info(f"  Losing Trade for {report_item['symbol']}: Entered {report_item['entry_side']} @ {report_item['entry_price']:.2f}, Exited @ {report_item['exit_price']:.2f}, P&L: {report_item['net_pnl']:.2f}")
            logger.info(f"    AI Entry Reasoning: {ai_reasoning_at_entry[:200]}...") # Log snippet
            if sub_signals_at_entry:
                logger.info(f"    Sub-signals at entry (first 2): {json.dumps(list(sub_signals_at_entry.values())[:2], indent=1, default=str)}")


            failure_analysis_report.append(report_item)
        
        logger.info("--- END OF FAILED TRADES ANALYSIS REPORT ---")
        # In a real system, this report could be written to a file or a database for review.
        return pd.DataFrame(failure_analysis_report)


# Example Usage (typically called after a backtest run)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("--- Testing BacktestTradeAnalyzer ---")

    # Create dummy trade log data (as if from PortfolioManager)
    # Ensure 'pnl' is calculated for paired trades, or 'net_pnl' after brokerage
    # Add conceptual AI decision details if available
    dummy_trades_list = [
        {'timestamp': pd.to_datetime('2023-01-02 09:30'), 'instrument_token': 101, 'trading_symbol': 'STOCK_A', 'side': 'BUY', 'quantity': 10, 'price': 100.0, 'brokerage': 5, 'ai_decision_details': {"reasoning": "AI BUY reason A"}, 'contributing_sub_agent_signals': {"RSI_AI": {"signal":"BUY"}}},
        {'timestamp': pd.to_datetime('2023-01-02 10:00'), 'instrument_token': 101, 'trading_symbol': 'STOCK_A', 'side': 'SELL', 'quantity': 10, 'price': 105.0, 'brokerage': 5, 'exit_reason': 'Profit Target'}, # Profit
        
        {'timestamp': pd.to_datetime('2023-01-02 10:30'), 'instrument_token': 102, 'trading_symbol': 'STOCK_B', 'side': 'SELL', 'quantity': 20, 'price': 200.0, 'brokerage': 10, 'ai_decision_details': {"reasoning": "AI SELL reason B"}},
        {'timestamp': pd.to_datetime('2023-01-02 11:00'), 'instrument_token': 102, 'trading_symbol': 'STOCK_B', 'side': 'BUY', 'quantity': 20, 'price': 205.0, 'brokerage': 10, 'exit_reason': 'Stop Loss'}, # Loss

        {'timestamp': pd.to_datetime('2023-01-02 11:30'), 'instrument_token': 101, 'trading_symbol': 'STOCK_A', 'side': 'BUY', 'quantity': 5, 'price': 102.0, 'brokerage': 2.5, 'ai_decision_details': {"reasoning": "AI BUY reason A2"}},
        {'timestamp': pd.to_datetime('2023-01-02 12:00'), 'instrument_token': 101, 'trading_symbol': 'STOCK_A', 'side': 'SELL', 'quantity': 5, 'price': 100.0, 'brokerage': 2.5, 'exit_reason': 'Stop Loss'}, # Loss
    ]
    trade_log_df_test = pd.DataFrame(dummy_trades_list)

    # Dummy portfolio summary (normally from PortfolioManager)
    dummy_summary = {
        "initial_capital": 100000, "final_portfolio_value": 99000, # Example
        # "win_rate_percentage": 33.33 # This would be calculated by the analyzer
    }

    analyzer = BacktestTradeAnalyzer(trade_log_df_test, dummy_summary)
    
    logger.info("\nPaired Trades:")
    if not analyzer.paired_trades_df.empty:
        logger.info(analyzer.paired_trades_df[['entry_timestamp', 'trading_symbol', 'entry_side', 'entry_price', 'exit_price', 'net_pnl']])
    else:
        logger.info("No trades were paired.")

    stats, winning_df, losing_df = analyzer.calculate_win_loss_stats()
    
    # Simulate low accuracy to trigger detailed report
    # To properly test this, the portfolio_summary would need the win_rate, or it's calculated internally.
    # Let's assume the calculated win_rate is low for this test.
    logger.info("\n--- Simulating Low Accuracy for Failure Report ---")
    # For testing, we can temporarily modify the win_rate used by analyze_failed_trades_report
    # or ensure our dummy data results in low win rate.
    # The current dummy data has 1 win, 2 losses from paired trades. Win rate = 33.33%
    
    failed_trades_report_df = analyzer.analyze_failed_trades_report(losing_df, accuracy_threshold=40.0) # Set threshold low to trigger
    if failed_trades_report_df is not None and not failed_trades_report_df.empty:
        logger.info("\nGenerated Failed Trades Report DataFrame (first few rows):")
        logger.info(failed_trades_report_df.head().to_string())
    elif failed_trades_report_df is not None: # Empty dataframe
        logger.info("Failed trades report generated, but it's empty (no losing trades or other issue).")

