"""
Performance Metrics Module for Backtesting

Calculates professional-grade quant metrics including risk-adjusted returns,
drawdown analysis, regime-based performance, and trade quality metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    
    # Risk-Adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0  # Expected Shortfall
    
    # Market Exposure
    beta: float = 0.0
    alpha: float = 0.0
    
    # Trade Quality
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_period: float = 0.0
    total_trades: int = 0
    turnover: float = 0.0
    
    # Regime Performance
    regime_metrics: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'returns': {
                'total_return': f"{self.total_return:.2%}",
                'cagr': f"{self.cagr:.2%}",
            },
            'risk_adjusted': {
                'sharpe_ratio': round(self.sharpe_ratio, 3),
                'sortino_ratio': round(self.sortino_ratio, 3),
                'calmar_ratio': round(self.calmar_ratio, 3),
                'information_ratio': round(self.information_ratio, 3),
            },
            'risk': {
                'volatility': f"{self.volatility:.2%}",
                'max_drawdown': f"{self.max_drawdown:.2%}",
                'avg_drawdown': f"{self.avg_drawdown:.2%}",
                'var_95': f"{self.var_95:.2%}",
                'var_99': f"{self.var_99:.2%}",
                'cvar_95': f"{self.cvar_95:.2%}",
            },
            'market_exposure': {
                'beta': round(self.beta, 3),
                'alpha': f"{self.alpha:.2%}",
            },
            'trade_quality': {
                'win_rate': f"{self.win_rate:.1%}",
                'profit_factor': round(self.profit_factor, 2),
                'avg_win': f"${self.avg_win:.2f}",
                'avg_loss': f"${self.avg_loss:.2f}",
                'avg_holding_period': f"{self.avg_holding_period:.1f} days",
                'total_trades': self.total_trades,
                'turnover': f"{self.turnover:.1%}",
            },
            'regime_performance': self.regime_metrics
        }


class PerformanceCalculator:
    """
    Calculates comprehensive performance metrics for backtesting.
    
    Follows industry-standard methodologies used by professional quant desks.
    """
    
    def __init__(self, risk_free_rate: float = 0.04, trading_days_per_year: int = 252):
        """
        Initialize calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 4% for current environment)
            trading_days_per_year: Trading days for annualization (252)
        """
        self.rf = risk_free_rate
        self.trading_days = trading_days_per_year
    
    def calculate_all_metrics(
        self,
        portfolio_values: pd.Series,
        benchmark_values: pd.Series,
        trades: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Args:
            portfolio_values: Daily portfolio values (DatetimeIndex)
            benchmark_values: Daily benchmark values (e.g., SPY)
            trades: DataFrame with columns [date, ticker, action, shares, price, pnl]
            regime_labels: Optional series mapping dates to regimes (bull/bear/sideways)
            
        Returns:
            PerformanceMetrics dataclass
        """
        metrics = PerformanceMetrics()
        
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        benchmark_returns = benchmark_values.pct_change().dropna()
        
        # Align returns
        returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
        
        if len(returns) < 2:
            return metrics
        
        # Return metrics
        metrics.total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        years = len(returns) / self.trading_days
        metrics.cagr = (1 + metrics.total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        metrics.volatility = returns.std() * np.sqrt(self.trading_days)
        
        # Sharpe Ratio
        excess_returns = returns.mean() - (self.rf / self.trading_days)
        metrics.sharpe_ratio = (excess_returns / returns.std()) * np.sqrt(self.trading_days) if returns.std() > 0 else 0
        
        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.trading_days) if len(downside_returns) > 0 else 0
        metrics.sortino_ratio = (metrics.cagr - self.rf) / downside_std if downside_std > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        metrics.max_drawdown = abs(drawdowns.min())
        metrics.avg_drawdown = abs(drawdowns[drawdowns < 0].mean()) if len(drawdowns[drawdowns < 0]) > 0 else 0
        
        # Calmar Ratio
        metrics.calmar_ratio = metrics.cagr / metrics.max_drawdown if metrics.max_drawdown > 0 else 0
        
        # VaR and CVaR
        metrics.var_95 = abs(np.percentile(returns, 5))
        metrics.var_99 = abs(np.percentile(returns, 1))
        metrics.cvar_95 = abs(returns[returns <= np.percentile(returns, 5)].mean())
        
        # Beta and Alpha (CAPM)
        if len(benchmark_returns) > 0 and benchmark_returns.std() > 0:
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            metrics.beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            benchmark_return = (1 + benchmark_returns).prod() ** (self.trading_days / len(benchmark_returns)) - 1
            expected_return = self.rf + metrics.beta * (benchmark_return - self.rf)
            metrics.alpha = metrics.cagr - expected_return
        
        # Information Ratio
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(self.trading_days)
        metrics.information_ratio = (active_returns.mean() * self.trading_days) / tracking_error if tracking_error > 0 else 0
        
        # Trade quality metrics
        if trades is not None and len(trades) > 0:
            metrics = self._calculate_trade_metrics(metrics, trades, portfolio_values)
        
        # Regime analysis
        if regime_labels is not None:
            metrics.regime_metrics = self._calculate_regime_metrics(returns, benchmark_returns, regime_labels)
        
        return metrics
    
    def _calculate_trade_metrics(
        self, 
        metrics: PerformanceMetrics, 
        trades: pd.DataFrame,
        portfolio_values: pd.Series
    ) -> PerformanceMetrics:
        """Calculate trade-level quality metrics."""
        
        # Filter completed trades (need SELL to realize P&L)
        if 'pnl' not in trades.columns:
            return metrics
            
        completed_trades = trades[trades['action'] == 'SELL'].copy()
        
        if len(completed_trades) == 0:
            return metrics
        
        metrics.total_trades = len(completed_trades)
        
        # Win rate
        winning_trades = completed_trades[completed_trades['pnl'] > 0]
        losing_trades = completed_trades[completed_trades['pnl'] < 0]
        
        metrics.win_rate = len(winning_trades) / len(completed_trades) if len(completed_trades) > 0 else 0
        
        # Average win/loss
        metrics.avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        metrics.avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Holding period (simplified - days between BUY and SELL)
        if 'holding_days' in trades.columns:
            metrics.avg_holding_period = trades['holding_days'].mean()
        
        # Turnover (total value traded / avg portfolio value)
        if 'amount' in trades.columns:
            total_traded = trades['amount'].sum()
            avg_portfolio = portfolio_values.mean()
            years = len(portfolio_values) / self.trading_days
            metrics.turnover = (total_traded / avg_portfolio / years) if avg_portfolio > 0 and years > 0 else 0
        
        return metrics
    
    def _calculate_regime_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        regime_labels: pd.Series
    ) -> Dict[str, Dict]:
        """Calculate performance split by market regime."""
        
        regime_metrics = {}
        
        for regime in regime_labels.unique():
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) < 5:
                continue
            
            # Calculate metrics for this regime
            regime_vol = regime_returns.std() * np.sqrt(self.trading_days)
            regime_sharpe = (
                (regime_returns.mean() - self.rf / self.trading_days) / regime_returns.std() 
                * np.sqrt(self.trading_days)
            ) if regime_returns.std() > 0 else 0
            
            cum_regime = (1 + regime_returns).cumprod()
            regime_max = cum_regime.cummax()
            regime_dd = ((cum_regime - regime_max) / regime_max).min()
            
            regime_metrics[regime] = {
                'days': len(regime_returns),
                'total_return': f"{((1 + regime_returns).prod() - 1):.2%}",
                'sharpe': round(regime_sharpe, 2),
                'volatility': f"{regime_vol:.2%}",
                'max_drawdown': f"{abs(regime_dd):.2%}"
            }
        
        return regime_metrics

    @staticmethod
    def classify_market_regimes(
        benchmark_prices: pd.Series,
        sma_period: int = 200,
        volatility_lookback: int = 30
    ) -> pd.Series:
        """
        Classify market regimes based on trend and volatility.
        
        Regimes:
        - 'bull': Price > SMA200, low volatility
        - 'bear': Price < SMA200, high volatility
        - 'crisis': Extreme volatility (> 2x average)
        - 'sideways': No clear trend
        
        Args:
            benchmark_prices: Benchmark price series (e.g., SPY)
            sma_period: Period for trend detection
            volatility_lookback: Days for volatility calculation
            
        Returns:
            Series of regime labels indexed by date
        """
        regimes = pd.Series(index=benchmark_prices.index, dtype=str)
        
        # Calculate indicators
        sma = benchmark_prices.rolling(window=sma_period).mean()
        returns = benchmark_prices.pct_change()
        volatility = returns.rolling(window=volatility_lookback).std() * np.sqrt(252)
        avg_volatility = volatility.mean()
        
        for date in benchmark_prices.index[sma_period:]:
            price = benchmark_prices.loc[date]
            sma_val = sma.loc[date]
            vol = volatility.loc[date]
            
            if pd.isna(vol) or pd.isna(sma_val):
                regimes.loc[date] = 'unknown'
                continue
            
            # Crisis: Extreme volatility
            if vol > 2 * avg_volatility:
                regimes.loc[date] = 'crisis'
            # Bull: Above SMA with moderate volatility
            elif price > sma_val and vol < 1.5 * avg_volatility:
                regimes.loc[date] = 'bull'
            # Bear: Below SMA
            elif price < sma_val:
                regimes.loc[date] = 'bear'
            # Sideways: Everything else
            else:
                regimes.loc[date] = 'sideways'
        
        return regimes.fillna('unknown')


def generate_performance_summary(metrics: PerformanceMetrics) -> str:
    """Generate human-readable performance summary."""
    
    summary = []
    summary.append("=" * 60)
    summary.append("BACKTEST PERFORMANCE SUMMARY")
    summary.append("=" * 60)
    
    summary.append("\n📈 RETURNS")
    summary.append(f"   Total Return: {metrics.total_return:.2%}")
    summary.append(f"   CAGR: {metrics.cagr:.2%}")
    
    summary.append("\n⚖️ RISK-ADJUSTED")
    summary.append(f"   Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    summary.append(f"   Sortino Ratio: {metrics.sortino_ratio:.3f}")
    summary.append(f"   Calmar Ratio: {metrics.calmar_ratio:.3f}")
    summary.append(f"   Information Ratio: {metrics.information_ratio:.3f}")
    
    summary.append("\n🎯 RISK")
    summary.append(f"   Volatility (Ann.): {metrics.volatility:.2%}")
    summary.append(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
    summary.append(f"   VaR (95%): {metrics.var_95:.2%}")
    summary.append(f"   CVaR (95%): {metrics.cvar_95:.2%}")
    
    summary.append("\n📊 MARKET EXPOSURE")
    summary.append(f"   Beta: {metrics.beta:.3f}")
    summary.append(f"   Alpha: {metrics.alpha:.2%}")
    
    summary.append("\n💼 TRADE QUALITY")
    summary.append(f"   Win Rate: {metrics.win_rate:.1%}")
    summary.append(f"   Profit Factor: {metrics.profit_factor:.2f}")
    summary.append(f"   Total Trades: {metrics.total_trades}")
    summary.append(f"   Annual Turnover: {metrics.turnover:.1%}")
    
    if metrics.regime_metrics:
        summary.append("\n🌍 REGIME PERFORMANCE")
        for regime, data in metrics.regime_metrics.items():
            summary.append(f"   {regime.upper()}: Sharpe={data['sharpe']}, Return={data['total_return']}, MaxDD={data['max_drawdown']}")
    
    summary.append("\n" + "=" * 60)
    
    return "\n".join(summary)
