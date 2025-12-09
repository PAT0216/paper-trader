"""
Risk Management Module for Paper Trader

Provides position sizing, portfolio constraints, and risk metrics to prevent
catastrophic losses and ensure prudent capital allocation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RiskLimits:
    """Configuration for risk management constraints."""
    max_position_pct: float = 0.15  # Max 15% per position
    max_sector_pct: float = 0.40    # Max 40% in any sector
    min_cash_buffer: float = 100.0  # Minimum cash reserve
    max_daily_var_pct: float = 0.025  # Max 2.5% Value at Risk
    volatility_lookback: int = 30   # Days for volatility calculation
    correlation_threshold: float = 0.7  # Reduce size if correlation > 70%


class RiskManager:
    """
    Manages portfolio-level risk through position sizing, constraint validation,
    and risk metric calculation.
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager with configurable limits.
        
        Args:
            risk_limits: RiskLimits dataclass with constraint parameters
        """
        self.limits = risk_limits or RiskLimits()
        
        # Sector mappings for S&P 500 tickers (simplified)
        self.sector_map = {
            # Indices
            'SPY': 'Index', 'QQQ': 'Index', 'IWM': 'Index', 'DIA': 'Index',
            # Tech
            'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
            'GOOGL': 'Technology', 'AMZN': 'Technology', 'META': 'Technology',
            'TSLA': 'Technology', 'AVGO': 'Technology', 'AMD': 'Technology',
            'CRM': 'Technology', 'NFLX': 'Technology',
            # Finance
            'JPM': 'Financials', 'V': 'Financials', 'MA': 'Financials',
            'BAC': 'Financials', 'BRK-B': 'Financials',
            # Consumer
            'HD': 'Consumer', 'COST': 'Consumer', 'WMT': 'Consumer',
            'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer', 'DIS': 'Consumer',
            # Healthcare
            'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'MRK': 'Healthcare',
            # Energy/Industrial
            'XOM': 'Energy', 'CVX': 'Energy', 'BA': 'Industrials', 'CAT': 'Industrials',
        }
    
    def calculate_position_size(
        self,
        ticker: str,
        current_price: float,
        available_cash: float,
        portfolio_value: float,
        historical_data: pd.DataFrame,
        current_holdings: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Tuple[int, str]:
        """
        Calculate optimal position size based on volatility, portfolio constraints,
        and risk limits.
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current price of the asset
            available_cash: Available cash for trading
            portfolio_value: Total portfolio value (cash + holdings)
            historical_data: OHLC data for volatility calculation
            current_holdings: Dict of {ticker: shares} for current positions
            current_prices: Dict of {ticker: price} for current prices
            
        Returns:
            Tuple of (shares_to_buy, reason) where reason explains the sizing logic
        """
        if available_cash < current_price:
            return 0, "Insufficient cash for even 1 share"
        
        # Step 1: Calculate position limit based on max position percentage
        max_position_value = portfolio_value * self.limits.max_position_pct
        max_shares_by_position_limit = int(max_position_value / current_price)
        
        # Step 2: Calculate volatility-adjusted position size
        volatility = self._calculate_volatility(historical_data)
        if volatility is None or volatility == 0:
            volatility_adjusted_shares = max_shares_by_position_limit
            vol_reason = "using default sizing"
        else:
            # Inverse volatility weighting: higher volatility = smaller position
            # Target volatility = 20% annualized (reasonable for equities)
            target_vol = 0.20
            vol_scalar = min(target_vol / volatility, 1.5)  # Cap at 1.5x adjustment
            volatility_adjusted_shares = int(max_shares_by_position_limit * vol_scalar)
            vol_reason = f"vol={volatility:.2%}, scalar={vol_scalar:.2f}"
        
        # Step 3: Check sector concentration limits
        sector = self.sector_map.get(ticker, 'Other')
        current_sector_exposure = self._calculate_sector_exposure(
            current_holdings, current_prices, sector
        )
        
        max_additional_sector_value = (
            portfolio_value * self.limits.max_sector_pct - current_sector_exposure
        )
        
        if max_additional_sector_value <= 0:
            return 0, f"Sector limit reached: {sector} already at {current_sector_exposure/portfolio_value:.1%}"
        
        max_shares_by_sector = int(max_additional_sector_value / current_price)
        
        # Step 4: Check correlation with existing holdings
        correlation_penalty = self._calculate_correlation_penalty(
            ticker, historical_data, current_holdings, current_prices, portfolio_value
        )
        
        # Step 5: Determine final position size (minimum of all constraints)
        max_shares_by_cash = int(available_cash / current_price)
        
        final_shares = min(
            max_shares_by_cash,
            max_shares_by_position_limit,
            volatility_adjusted_shares,
            max_shares_by_sector
        )
        
        # Apply correlation penalty
        final_shares = int(final_shares * (1 - correlation_penalty))
        
        reason = (
            f"Size: ${final_shares * current_price:.0f} "
            f"({final_shares * current_price / portfolio_value:.1%} of portfolio), "
            f"{vol_reason}"
        )
        
        if correlation_penalty > 0:
            reason += f", corr_penalty={correlation_penalty:.1%}"
        
        return final_shares, reason
    
    def _calculate_volatility(self, df: pd.DataFrame, lookback: Optional[int] = None) -> Optional[float]:
        """
        Calculate annualized volatility from historical OHLC data.
        
        Args:
            df: DataFrame with 'Close' column
            lookback: Number of days to use (default: from RiskLimits)
            
        Returns:
            Annualized volatility (std dev of returns) or None if insufficient data
        """
        lookback = lookback or self.limits.volatility_lookback
        
        if len(df) < lookback:
            return None
        
        recent_df = df.tail(lookback)
        returns = recent_df['Close'].pct_change().dropna()
        
        if len(returns) < 2:
            return None
        
        # Annualize daily volatility (sqrt(252 trading days))
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        return annualized_vol
    
    def _calculate_sector_exposure(
        self,
        holdings: Dict[str, float],
        prices: Dict[str, float],
        target_sector: str
    ) -> float:
        """
        Calculate total portfolio value in a specific sector.
        
        Args:
            holdings: Dict of {ticker: shares}
            prices: Dict of {ticker: price}
            target_sector: Sector name to calculate exposure for
            
        Returns:
            Total value in dollars for the target sector
        """
        sector_value = 0.0
        
        for ticker, shares in holdings.items():
            if self.sector_map.get(ticker, 'Other') == target_sector:
                price = prices.get(ticker, 0.0)
                sector_value += shares * price
        
        return sector_value
    
    def _calculate_correlation_penalty(
        self,
        ticker: str,
        ticker_data: pd.DataFrame,
        current_holdings: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> float:
        """
        Calculate position size penalty based on correlation with existing holdings.
        High correlation = reduce new position to maintain diversification.
        
        Args:
            ticker: New ticker to potentially buy
            ticker_data: Historical data for the new ticker
            current_holdings: Current positions
            current_prices: Current prices for all tickers
            portfolio_value: Total portfolio value
            
        Returns:
            Penalty factor (0.0 to 0.5) to reduce position size
        """
        # Simplified correlation penalty: check if same sector
        # In production, would calculate actual return correlation from historical data
        
        if not current_holdings:
            return 0.0  # No existing positions, no correlation concern
        
        ticker_sector = self.sector_map.get(ticker, 'Other')
        
        # Calculate weight of current holdings in same sector
        same_sector_value = self._calculate_sector_exposure(
            current_holdings, current_prices, ticker_sector
        )
        
        same_sector_weight = same_sector_value / portfolio_value if portfolio_value > 0 else 0
        
        # Apply penalty if sector already has significant exposure
        if same_sector_weight > 0.25:  # If sector is >25% of portfolio
            penalty = min((same_sector_weight - 0.25) / 0.25, 0.5)  # Up to 50% penalty
            return penalty
        
        return 0.0
    
    def calculate_portfolio_var(
        self,
        holdings: Dict[str, float],
        prices: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame],
        confidence: float = 0.95
    ) -> Optional[float]:
        """
        Calculate portfolio Value at Risk (VaR) using historical simulation.
        
        Args:
            holdings: Dict of {ticker: shares}
            prices: Dict of {ticker: current_price}
            historical_data: Dict of {ticker: OHLC DataFrame}
            confidence: Confidence level (default 95%)
            
        Returns:
            VaR in dollars (potential loss) or None if insufficient data
        """
        if not holdings:
            return 0.0
        
        # Calculate portfolio value
        portfolio_value = sum(shares * prices.get(ticker, 0) for ticker, shares in holdings.items())
        
        if portfolio_value == 0:
            return 0.0
        
        # Get daily returns for each holding
        portfolio_returns = []
        lookback = self.limits.volatility_lookback
        
        for ticker, shares in holdings.items():
            if ticker not in historical_data:
                continue
            
            df = historical_data[ticker].tail(lookback)
            if len(df) < 2:
                continue
            
            daily_returns = df['Close'].pct_change().dropna()
            position_value = shares * prices.get(ticker, 0)
            weight = position_value / portfolio_value
            
            if len(portfolio_returns) == 0:
                portfolio_returns = (daily_returns * weight).values
            else:
                # Align lengths (simple approach: trim to minimum)
                min_len = min(len(portfolio_returns), len(daily_returns))
                portfolio_returns = portfolio_returns[-min_len:] + (daily_returns.values[-min_len:] * weight)
        
        if len(portfolio_returns) == 0:
            return None
        
        # Calculate VaR at specified confidence level
        var_percentile = (1 - confidence) * 100
        var_return = np.percentile(portfolio_returns, var_percentile)
        var_dollars = abs(var_return * portfolio_value)
        
        return var_dollars
    
    def validate_trade(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        current_holdings: Dict[str, float],
        current_prices: Dict[str, float],
        cash_balance: float,
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """
        Pre-trade validation to ensure trade complies with risk limits.
        
        Args:
            ticker: Stock ticker
            action: "BUY" or "SELL"
            shares: Number of shares
            price: Execution price
            current_holdings: Current positions
            current_prices: Current market prices
            cash_balance: Available cash
            portfolio_value: Total portfolio value
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if action == "BUY":
            trade_value = shares * price
            
            # Check 1: Sufficient cash
            if trade_value > cash_balance:
                return False, f"Insufficient cash: need ${trade_value:.2f}, have ${cash_balance:.2f}"
            
            # Check 2: Position size limit
            position_pct = trade_value / portfolio_value
            if position_pct > self.limits.max_position_pct:
                return False, f"Position too large: {position_pct:.1%} > {self.limits.max_position_pct:.1%} limit"
            
            # Check 3: Sector concentration
            sector = self.sector_map.get(ticker, 'Other')
            sector_exposure = self._calculate_sector_exposure(current_holdings, current_prices, sector)
            new_sector_exposure = sector_exposure + trade_value
            sector_pct = new_sector_exposure / portfolio_value
            
            if sector_pct > self.limits.max_sector_pct:
                return False, f"Sector limit exceeded: {sector} would be {sector_pct:.1%} > {self.limits.max_sector_pct:.1%}"
            
            # Check 4: Min cash buffer
            remaining_cash = cash_balance - trade_value
            if remaining_cash < self.limits.min_cash_buffer:
                return False, f"Min cash buffer violated: ${remaining_cash:.2f} < ${self.limits.min_cash_buffer:.2f}"
            
            return True, "Trade validated"
        
        elif action == "SELL":
            # Validate we actually own the shares
            current_shares = current_holdings.get(ticker, 0)
            if shares > current_shares:
                return False, f"Cannot sell {shares} shares, only own {current_shares}"
            
            return True, "Trade validated"
        
        else:
            return False, f"Unknown action: {action}"
    
    def get_sector_exposure_summary(
        self,
        holdings: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate exposure percentage for each sector.
        
        Returns:
            Dict of {sector: percentage_of_portfolio}
        """
        sector_values = {}
        
        for ticker, shares in holdings.items():
            sector = self.sector_map.get(ticker, 'Other')
            price = prices.get(ticker, 0.0)
            value = shares * price
            
            sector_values[sector] = sector_values.get(sector, 0.0) + value
        
        if portfolio_value == 0:
            return {}
        
        return {sector: value / portfolio_value for sector, value in sector_values.items()}
