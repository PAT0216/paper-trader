"""
Backtesting Engine for Paper Trader

Event-driven backtester that simulates trading strategy performance
over historical data with realistic transaction costs and risk controls.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os

from src.backtesting.performance import (
    PerformanceCalculator, 
    PerformanceMetrics,
    generate_performance_summary
)
from src.backtesting.costs import TransactionCostModel, CostTracker, CostConfig
from src.trading.risk_manager import RiskManager, RiskLimits, DrawdownController


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str = "2017-01-01"
    end_date: str = "2024-12-31"
    initial_cash: float = 100000.0
    benchmark_ticker: str = "SPY"
    
    # Risk settings
    max_position_pct: float = 0.15
    max_sector_pct: float = 0.30  # Updated Phase 7
    min_cash_buffer: float = 200.0  # Updated Phase 7
    
    # Phase 7: Stop-loss and drawdown controls
    # A/B tested with diverse stocks - 15% is balanced default
    stop_loss_pct: float = 0.15  # Sell if position drops 15% from entry
    drawdown_warning: float = 0.15  # 50% position reduction at -15%
    drawdown_halt: float = 0.20  # No new buys at -20%
    drawdown_liquidate: float = 0.25  # Force liquidation at -25%
    use_stop_loss: bool = True  # Enable stop-loss
    use_drawdown_control: bool = True  # Enable drawdown control
    
    # Cost settings
    slippage_bps: float = 5.0
    commission_per_share: float = 0.0
    
    # Execution settings
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    use_risk_manager: bool = True
    
    # Output settings
    use_dated_folders: bool = False  # Save to dated subdirectories

    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'dates': {'start': self.start_date, 'end': self.end_date},
            'initial_cash': self.initial_cash,
            'benchmark': self.benchmark_ticker,
            'risk': {
                'max_position_pct': f"{self.max_position_pct:.0%}",
                'max_sector_pct': f"{self.max_sector_pct:.0%}",
            },
            'costs': {
                'slippage_bps': self.slippage_bps,
                'commission_per_share': self.commission_per_share,
            },
            'rebalance_frequency': self.rebalance_frequency,
        }


@dataclass
class BacktestPosition:
    """Represents a position in the portfolio."""
    ticker: str
    shares: float
    avg_cost: float
    entry_date: pd.Timestamp
    
    def market_value(self, price: float) -> float:
        """Calculate current market value."""
        return self.shares * price
    
    def unrealized_pnl(self, price: float) -> float:
        """Calculate unrealized P&L."""
        return (price - self.avg_cost) * self.shares


@dataclass
class BacktestTrade:
    """Represents a completed trade."""
    date: pd.Timestamp
    ticker: str
    action: str
    shares: float
    price: float
    execution_price: float
    cost: float
    pnl: float = 0.0
    holding_days: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'date': self.date.strftime('%Y-%m-%d'),
            'ticker': self.ticker,
            'action': self.action,
            'shares': self.shares,
            'price': self.price,
            'execution_price': self.execution_price,
            'cost': self.cost,
            'pnl': self.pnl,
            'holding_days': self.holding_days
        }


class BacktestPortfolio:
    """
    Simulates portfolio state during backtesting.
    """
    
    def __init__(self, initial_cash: float, risk_manager: Optional[RiskManager] = None):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash amount
            risk_manager: Optional RiskManager for position sizing
        """
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[BacktestTrade] = []
        self.risk_manager = risk_manager
        self.cost_tracker = CostTracker()
        
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        holdings_value = sum(
            pos.market_value(prices.get(ticker, pos.avg_cost))
            for ticker, pos in self.positions.items()
        )
        return self.cash + holdings_value
    
    def get_holdings(self) -> Dict[str, float]:
        """Get current holdings as {ticker: shares}."""
        return {ticker: pos.shares for ticker, pos in self.positions.items()}
    
    def execute_trade(
        self,
        date: pd.Timestamp,
        ticker: str,
        action: str,
        shares: float,
        price: float,
        avg_daily_volume: Optional[float] = None
    ) -> Optional[BacktestTrade]:
        """
        Execute a trade with transaction costs.
        
        Args:
            date: Trade date
            ticker: Stock ticker
            action: 'BUY' or 'SELL'
            shares: Number of shares
            price: Market price
            avg_daily_volume: ADV for market impact (optional)
            
        Returns:
            BacktestTrade if executed, None if rejected
        """
        if shares <= 0:
            return None
        
        # Calculate execution price with costs
        execution_price, cost = self.cost_tracker.record_trade(
            date, ticker, action, int(shares), price, avg_daily_volume
        )
        
        trade = BacktestTrade(
            date=date,
            ticker=ticker,
            action=action,
            shares=shares,
            price=price,
            execution_price=execution_price,
            cost=cost
        )
        
        if action == 'BUY':
            trade_value = shares * execution_price
            
            if self.cash < trade_value:
                return None  # Insufficient cash
            
            self.cash -= trade_value
            
            if ticker in self.positions:
                # Average into position
                old_pos = self.positions[ticker]
                total_shares = old_pos.shares + shares
                avg_cost = (
                    (old_pos.shares * old_pos.avg_cost + shares * execution_price) 
                    / total_shares
                )
                self.positions[ticker] = BacktestPosition(
                    ticker=ticker,
                    shares=total_shares,
                    avg_cost=avg_cost,
                    entry_date=old_pos.entry_date
                )
            else:
                self.positions[ticker] = BacktestPosition(
                    ticker=ticker,
                    shares=shares,
                    avg_cost=execution_price,
                    entry_date=date
                )
                
        elif action == 'SELL':
            if ticker not in self.positions:
                return None  # No position to sell
            
            position = self.positions[ticker]
            
            if shares > position.shares:
                shares = position.shares  # Sell what we have
            
            trade_value = shares * execution_price
            self.cash += trade_value
            
            # Calculate realized P&L
            trade.pnl = (execution_price - position.avg_cost) * shares - cost
            trade.holding_days = (date - position.entry_date).days
            
            # Update position
            remaining = position.shares - shares
            if remaining <= 0:
                del self.positions[ticker]
            else:
                self.positions[ticker] = BacktestPosition(
                    ticker=ticker,
                    shares=remaining,
                    avg_cost=position.avg_cost,
                    entry_date=position.entry_date
                )
        
        self.trades.append(trade)
        return trade
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])


class Backtester:
    """
    Event-driven backtesting engine.
    
    Simulates strategy execution over historical data with:
    - Realistic transaction costs (slippage, commission, market impact)
    - Risk management integration
    - Regime-based performance analysis
    - Comprehensive metrics calculation
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.
        
        Args:
            config: BacktestConfig with simulation parameters
        """
        self.config = config or BacktestConfig()
        
        # Initialize cost model
        cost_config = CostConfig(
            slippage_bps=self.config.slippage_bps,
            commission_per_share=self.config.commission_per_share
        )
        self.cost_model = TransactionCostModel(cost_config)
        
        # Initialize risk manager
        if self.config.use_risk_manager:
            risk_limits = RiskLimits(
                max_position_pct=self.config.max_position_pct,
                max_sector_pct=self.config.max_sector_pct,
                min_cash_buffer=self.config.min_cash_buffer
            )
            self.risk_manager = RiskManager(risk_limits)
        else:
            self.risk_manager = None
        
        # Performance calculator
        self.perf_calc = PerformanceCalculator()
        
        # Results storage
        self.portfolio_values = []
        self.benchmark_values = []
        self.dates = []
        
        # Phase 7: Drawdown controller
        if self.config.use_drawdown_control:
            self.drawdown_controller = DrawdownController(
                warning_threshold=self.config.drawdown_warning,
                halt_threshold=self.config.drawdown_halt,
                liquidate_threshold=self.config.drawdown_liquidate
            )
        else:
            self.drawdown_controller = None
        
    def run(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signal_generator: Callable[[str, pd.DataFrame, Dict], str],
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Tuple[PerformanceMetrics, pd.DataFrame, Dict]:
        """
        Run backtest over historical data.
        
        Args:
            data_dict: Dict of {ticker: OHLCV DataFrame} with DatetimeIndex
            signal_generator: Function(ticker, df_up_to_date, portfolio_state) -> 'BUY'/'SELL'/'HOLD'
            benchmark_data: Optional benchmark OHLCV data (e.g., SPY)
            
        Returns:
            Tuple of (PerformanceMetrics, trades_df, summary_dict)
        """
        # Get all unique dates across tickers
        # Normalize to tz-naive for consistent comparison
        all_dates = set()
        for ticker, df in data_dict.items():
            # Convert to tz-naive if tz-aware
            if df.index.tz is not None:
                data_dict[ticker] = df.copy()
                data_dict[ticker].index = df.index.tz_localize(None)
            all_dates.update(data_dict[ticker].index.tolist())
        
        # Also normalize benchmark data
        if benchmark_data is not None and benchmark_data.index.tz is not None:
            benchmark_data = benchmark_data.copy()
            benchmark_data.index = benchmark_data.index.tz_localize(None)
        
        all_dates = sorted(all_dates)
        
        # Filter to config date range (tz-naive)
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        trading_dates = [d for d in all_dates if start <= d <= end]
        
        if len(trading_dates) == 0:
            raise ValueError(f"No trading dates in range {start} to {end}")
        
        print(f"Running backtest: {trading_dates[0].strftime('%Y-%m-%d')} to {trading_dates[-1].strftime('%Y-%m-%d')}")
        print(f"Trading days: {len(trading_dates)}")
        print(f"Tickers: {len(data_dict)}")
        
        # Initialize portfolio
        portfolio = BacktestPortfolio(
            initial_cash=self.config.initial_cash,
            risk_manager=self.risk_manager
        )
        portfolio.cost_tracker.cost_model = self.cost_model
        
        # Get benchmark prices if available
        benchmark_prices = None
        if benchmark_data is not None:
            benchmark_prices = benchmark_data['Close']
        
        # Track portfolio value over time
        portfolio_values = []
        benchmark_values = []
        dates_tracked = []
        
        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(trading_dates)
        
        # Main simulation loop
        for i, date in enumerate(trading_dates):
            # Get current prices
            current_prices = {}
            for ticker, df in data_dict.items():
                if date in df.index:
                    current_prices[ticker] = df.loc[date, 'Close']
            
            # Record portfolio value
            portfolio_value = portfolio.get_total_value(current_prices)
            portfolio_values.append(portfolio_value)
            dates_tracked.append(date)
            
            if benchmark_prices is not None and date in benchmark_prices.index:
                benchmark_values.append(benchmark_prices.loc[date])
            
            # Check if rebalance day
            if date not in rebalance_dates:
                continue
            
            # ==================== PHASE 7: STOP-LOSS CHECK ====================
            if self.config.use_stop_loss:
                stop_losses_triggered = []
                for ticker, position in list(portfolio.positions.items()):
                    current_price = current_prices.get(ticker)
                    if current_price is None:
                        continue
                    
                    loss_pct = (current_price - position.avg_cost) / position.avg_cost
                    if loss_pct < -self.config.stop_loss_pct:
                        stop_losses_triggered.append((ticker, current_price, position))
                
                # Execute stop-loss sells
                for ticker, price, position in stop_losses_triggered:
                    trade = portfolio.execute_trade(
                        date=date,
                        ticker=ticker,
                        action='SELL',
                        shares=position.shares,
                        price=price
                    )
                    if trade:
                        available_cash = portfolio.cash - self.config.min_cash_buffer
            # ==================== END STOP-LOSS ====================
            
            # ==================== PHASE 7: DRAWDOWN CHECK ====================
            position_multiplier = 1.0
            if self.drawdown_controller:
                self.drawdown_controller.update(portfolio_value)
                position_multiplier = self.drawdown_controller.get_position_multiplier()
                
                # Emergency liquidation
                if self.drawdown_controller.should_liquidate():
                    for ticker, position in list(portfolio.positions.items()):
                        price = current_prices.get(ticker)
                        if price:
                            portfolio.execute_trade(
                                date=date,
                                ticker=ticker,
                                action='SELL',
                                shares=position.shares // 2,  # Sell 50%
                                price=price
                            )
            # ==================== END DRAWDOWN ====================
            
            # Generate signals for each ticker
            # NOTE: Signal is generated based on data up to current bar's Close
            # But execution will happen at NEXT bar's Open (see below)
            signals = {}
            expected_returns = {}  # Track expected returns for priority ranking
            
            for ticker, df in data_dict.items():
                if date not in df.index:
                    continue
                
                # Get data up to current date (no look-ahead in features)
                df_to_date = df.loc[:date]
                
                if len(df_to_date) < 50:  # Need enough data for indicators
                    continue
                
                # Generate signal
                portfolio_state = {
                    'holdings': portfolio.get_holdings(),
                    'cash': portfolio.cash,
                    'total_value': portfolio_value,
                    'prices': current_prices
                }
                
                try:
                    result = signal_generator(ticker, df_to_date, portfolio_state)
                    # Support both old (signal only) and new (signal, expected_return) interface
                    if isinstance(result, tuple):
                        signal, exp_ret = result
                    else:
                        signal = result
                        exp_ret = 0.0  # Default for old interface
                    signals[ticker] = signal
                    expected_returns[ticker] = exp_ret
                except Exception as e:
                    signals[ticker] = 'HOLD'
                    expected_returns[ticker] = 0.0
            
            # ============================================================
            # EXECUTION: Uses NEXT bar's Open price (T+1 Open)
            # This is realistic: signal at T close, execute at T+1 open
            # ============================================================
            
            # Get next trading day for execution prices
            next_day_idx = i + 1
            if next_day_idx >= len(trading_dates):
                continue  # No next day to execute
            
            next_date = trading_dates[next_day_idx]
            
            # Get next day's OPEN prices for execution
            execution_prices = {}
            for ticker, df in data_dict.items():
                if next_date in df.index and 'Open' in df.columns:
                    execution_prices[ticker] = df.loc[next_date, 'Open']
                elif next_date in df.index:
                    execution_prices[ticker] = df.loc[next_date, 'Close']  # Fallback
            
            # Execute SELL signals first (to free up cash)
            for ticker, signal in signals.items():
                if signal == 'SELL' and ticker in portfolio.positions:
                    position = portfolio.positions[ticker]
                    price = execution_prices.get(ticker)
                    if price:
                        portfolio.execute_trade(
                            date=next_date,  # Trade on next day
                            ticker=ticker,
                            action='SELL',
                            shares=position.shares,
                            price=price
                        )
            
            # Execute BUY signals with risk-adjusted sizing
            # CRITICAL: Sort by expected return (highest first) for priority allocation
            buy_candidates = [
                (t, expected_returns.get(t, 0.0)) 
                for t, s in signals.items() 
                if s == 'BUY'
            ]
            buy_candidates.sort(key=lambda x: x[1], reverse=True)  # Highest expected return first
            
            if buy_candidates and portfolio.cash > self.config.min_cash_buffer:
                available_cash = portfolio.cash - self.config.min_cash_buffer
                
                for ticker, exp_return in buy_candidates:
                    price = execution_prices.get(ticker)
                    if not price or price <= 0:
                        continue
                    
                    # Calculate position size
                    if self.risk_manager:
                        ticker_df = data_dict.get(ticker)
                        if ticker_df is not None:
                            df_to_date = ticker_df.loc[:date]  # Use signal date for vol calc
                            shares, _ = self.risk_manager.calculate_position_size(
                                ticker=ticker,
                                current_price=price,
                                available_cash=available_cash,
                                portfolio_value=portfolio_value,
                                historical_data=df_to_date,
                                current_holdings=portfolio.get_holdings(),
                                current_prices=execution_prices
                            )
                        else:
                            shares = int(available_cash * 0.1 / price)
                    else:
                        # Equal weight fallback
                        budget = available_cash / max(len(buy_candidates), 1)
                        shares = int(budget / price)
                    
                    if shares > 0:
                        # Apply drawdown position multiplier
                        if position_multiplier < 1.0:
                            shares = int(shares * position_multiplier)
                        
                        if shares > 0:
                            trade = portfolio.execute_trade(
                                date=next_date,  # Trade on next day
                                ticker=ticker,
                                action='BUY',
                                shares=shares,
                                price=price
                            )
                        
                        if trade:
                            available_cash = portfolio.cash - self.config.min_cash_buffer
            
            # Progress indicator
            if i % 100 == 0:
                print(f"  Day {i}/{len(trading_dates)}: Portfolio = ${portfolio_value:,.2f}")
        
        # Create results
        portfolio_series = pd.Series(portfolio_values, index=dates_tracked)
        
        if benchmark_values:
            # Normalize benchmark to same starting value
            benchmark_series = pd.Series(benchmark_values, index=dates_tracked[:len(benchmark_values)])
            benchmark_series = benchmark_series / benchmark_series.iloc[0] * self.config.initial_cash
        else:
            benchmark_series = pd.Series(dtype=float)
        
        # Calculate regime labels
        regime_labels = None
        if benchmark_data is not None and len(benchmark_data) > 200:
            regime_labels = PerformanceCalculator.classify_market_regimes(
                benchmark_data['Close']
            )
            # Align with portfolio dates
            regime_labels = regime_labels.reindex(dates_tracked).fillna('unknown')
        
        # Calculate performance metrics
        trades_df = portfolio.get_trades_df()
        
        metrics = self.perf_calc.calculate_all_metrics(
            portfolio_values=portfolio_series,
            benchmark_values=benchmark_series,
            trades=trades_df,
            regime_labels=regime_labels
        )
        
        # Build summary
        summary = {
            'config': self.config.to_dict(),
            'results': {
                'final_value': portfolio_series.iloc[-1],
                'total_return': metrics.total_return,
                'sharpe': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'total_trades': len(trades_df),
            },
            'costs': portfolio.cost_tracker.get_summary(),
            'metrics': metrics.to_dict()
        }
        
        # Store for later access
        self.portfolio_values = portfolio_series
        self.benchmark_values = benchmark_series
        self.dates = dates_tracked
        
        return metrics, trades_df, summary
    
    def _get_rebalance_dates(self, dates: List[pd.Timestamp]) -> set:
        """Get dates when rebalancing should occur."""
        if self.config.rebalance_frequency == 'daily':
            return set(dates)
        
        rebalance_dates = set()
        
        if self.config.rebalance_frequency == 'weekly':
            # Every Monday (or first trading day of week)
            current_week = None
            for d in dates:
                week = d.isocalendar()[1]
                if week != current_week:
                    rebalance_dates.add(d)
                    current_week = week
                    
        elif self.config.rebalance_frequency == 'monthly':
            # First trading day of each month
            current_month = None
            for d in dates:
                if d.month != current_month:
                    rebalance_dates.add(d)
                    current_month = d.month
        
        return rebalance_dates
    
    def generate_report(self, metrics: PerformanceMetrics, output_dir: str = "results"):
        """
        Generate backtest report files.
        
        Args:
            metrics: Calculated PerformanceMetrics
            output_dir: Directory to save reports
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Text summary
        summary_text = generate_performance_summary(metrics)
        with open(os.path.join(output_dir, "backtest_summary.txt"), "w") as f:
            f.write(summary_text)
        
        # JSON metrics
        with open(os.path.join(output_dir, "backtest_metrics.json"), "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        print(f"Reports saved to {output_dir}/")
        print(summary_text)


def create_ml_signal_generator(predictor, threshold_buy: float = 0.55, threshold_sell: float = 0.45):
    """
    Create a signal generator that uses the ML predictor.
    
    Args:
        predictor: Predictor instance with predict() method
        threshold_buy: Probability threshold for BUY signal
        threshold_sell: Probability threshold for SELL signal
        
    Returns:
        Signal generator function that returns (signal, expected_return)
    """
    def generate_signal(ticker: str, df: pd.DataFrame, portfolio_state: Dict) -> tuple:
        try:
            prob = predictor.predict(df)
            
            if prob > threshold_buy:
                signal = 'BUY'
            elif prob < threshold_sell:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Return both signal and expected return for priority ranking
            return (signal, prob)
        except Exception:
            return ('HOLD', 0.0)
    
    return generate_signal


def create_simple_signal_generator():
    """
    Create a simple SMA crossover signal generator for testing.
    
    Returns:
        Signal generator function
    """
    def generate_signal(ticker: str, df: pd.DataFrame, portfolio_state: Dict) -> str:
        if len(df) < 200:
            return 'HOLD'
        
        close = df['Close']
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        
        current_price = close.iloc[-1]
        current_sma50 = sma_50.iloc[-1]
        current_sma200 = sma_200.iloc[-1]
        
        # Already own?
        has_position = ticker in portfolio_state.get('holdings', {})
        
        # Golden cross: BUY
        if current_sma50 > current_sma200 and not has_position:
            return 'BUY'
        # Death cross: SELL
        elif current_sma50 < current_sma200 and has_position:
            return 'SELL'
        else:
            return 'HOLD'
    
    return generate_signal
