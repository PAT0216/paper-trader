"""
Transaction Cost Module for Backtesting

Models realistic trading costs including slippage, commissions, and market impact.
Uses retail-realistic assumptions based on modern discount brokerage environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CostConfig:
    """Configuration for transaction cost modeling."""
    # Commission (most retail brokers are now zero)
    commission_per_share: float = 0.0  # $0 for most brokers now
    min_commission: float = 0.0  # Minimum per trade
    
    # Slippage (bid-ask spread impact)
    slippage_bps: float = 5.0  # 5 basis points (0.05%)
    
    # Market impact (for large orders)
    market_impact_enabled: bool = True
    market_impact_coefficient: float = 0.1  # Sqrt law coefficient
    avg_daily_volume_threshold: float = 0.005  # 0.5% of ADV triggers impact
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.slippage_bps >= 0, "Slippage cannot be negative"
        assert self.commission_per_share >= 0, "Commission cannot be negative"


class TransactionCostModel:
    """
    Models realistic transaction costs for backtesting.
    
    Components:
    1. Commission: Per-share or per-trade fees
    2. Slippage: Bid-ask spread impact (5 bps typical for liquid stocks)
    3. Market Impact: Price movement from large orders (sqrt law)
    
    References:
    - Almgren & Chriss (2001) for market impact modeling
    - Typical retail spreads: 1-10 bps for liquid large caps
    """
    
    def __init__(self, config: Optional[CostConfig] = None):
        """
        Initialize cost model.
        
        Args:
            config: CostConfig with cost parameters (defaults to retail assumptions)
        """
        self.config = config or CostConfig()
    
    def calculate_execution_price(
        self,
        action: str,
        price: float,
        shares: int,
        avg_daily_volume: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate execution price after costs.
        
        Args:
            action: 'BUY' or 'SELL'
            price: Quoted/closing price
            shares: Number of shares
            avg_daily_volume: Average daily volume for market impact (optional)
            
        Returns:
            Tuple of (execution_price, total_cost)
        """
        if shares <= 0:
            return price, 0.0
        
        # 1. Slippage (always applies)
        slippage_pct = self.config.slippage_bps / 10000  # Convert bps to decimal
        
        # BUY: Pay more (ask side), SELL: Receive less (bid side)
        if action == 'BUY':
            slippage_impact = price * slippage_pct
        else:
            slippage_impact = -price * slippage_pct
        
        # 2. Market impact (for large orders)
        market_impact = 0.0
        if self.config.market_impact_enabled and avg_daily_volume is not None and avg_daily_volume > 0:
            participation_rate = shares / avg_daily_volume
            
            if participation_rate > self.config.avg_daily_volume_threshold:
                # Square root market impact model
                market_impact = (
                    self.config.market_impact_coefficient 
                    * np.sqrt(participation_rate) 
                    * price
                )
                
                if action == 'BUY':
                    market_impact = abs(market_impact)  # Pay more
                else:
                    market_impact = -abs(market_impact)  # Receive less
        
        # 3. Commission
        commission = self.config.commission_per_share * shares
        commission = max(commission, self.config.min_commission)
        
        # Calculate execution price
        if action == 'BUY':
            execution_price = price + slippage_impact + market_impact
            total_cost = (slippage_impact + market_impact) * shares + commission
        else:
            execution_price = price + slippage_impact + market_impact  # Both negative for SELL
            total_cost = abs(slippage_impact + market_impact) * shares + commission
        
        return execution_price, total_cost
    
    def calculate_trade_costs(
        self,
        trade_value: float,
        shares: int,
        avg_daily_volume: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Break down costs for a trade.
        
        Args:
            trade_value: Total dollar value of trade
            shares: Number of shares
            avg_daily_volume: ADV for market impact (optional)
            
        Returns:
            Dictionary with cost breakdown
        """
        price = trade_value / shares if shares > 0 else 0
        
        # Slippage
        slippage_cost = trade_value * (self.config.slippage_bps / 10000)
        
        # Commission
        commission = self.config.commission_per_share * shares
        commission = max(commission, self.config.min_commission)
        
        # Market impact
        market_impact_cost = 0.0
        if self.config.market_impact_enabled and avg_daily_volume is not None and avg_daily_volume > 0:
            participation_rate = shares / avg_daily_volume
            if participation_rate > self.config.avg_daily_volume_threshold:
                market_impact_cost = (
                    self.config.market_impact_coefficient 
                    * np.sqrt(participation_rate) 
                    * trade_value
                )
        
        total_cost = slippage_cost + commission + market_impact_cost
        
        return {
            'slippage': slippage_cost,
            'commission': commission,
            'market_impact': market_impact_cost,
            'total': total_cost,
            'cost_bps': (total_cost / trade_value * 10000) if trade_value > 0 else 0
        }


class CostTracker:
    """
    Tracks cumulative transaction costs during backtesting.
    """
    
    def __init__(self, cost_model: Optional[TransactionCostModel] = None):
        """
        Initialize tracker.
        
        Args:
            cost_model: TransactionCostModel instance
        """
        self.cost_model = cost_model or TransactionCostModel()
        self.total_slippage = 0.0
        self.total_commission = 0.0
        self.total_market_impact = 0.0
        self.trade_count = 0
        self.total_volume_traded = 0.0
        self.cost_history = []
    
    def record_trade(
        self,
        date: pd.Timestamp,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        avg_daily_volume: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Record a trade and calculate costs.
        
        Args:
            date: Trade date
            ticker: Stock ticker
            action: 'BUY' or 'SELL'
            shares: Number of shares
            price: Quoted price
            avg_daily_volume: ADV for market impact
            
        Returns:
            Tuple of (execution_price, total_cost)
        """
        trade_value = shares * price
        
        costs = self.cost_model.calculate_trade_costs(trade_value, shares, avg_daily_volume)
        execution_price, total_cost = self.cost_model.calculate_execution_price(
            action, price, shares, avg_daily_volume
        )
        
        # Update totals
        self.total_slippage += costs['slippage']
        self.total_commission += costs['commission']
        self.total_market_impact += costs['market_impact']
        self.trade_count += 1
        self.total_volume_traded += trade_value
        
        # Record history
        self.cost_history.append({
            'date': date,
            'ticker': ticker,
            'action': action,
            'shares': shares,
            'price': price,
            'execution_price': execution_price,
            'slippage': costs['slippage'],
            'commission': costs['commission'],
            'market_impact': costs['market_impact'],
            'total_cost': costs['total'],
            'cost_bps': costs['cost_bps']
        })
        
        return execution_price, total_cost
    
    def get_summary(self) -> Dict:
        """Get cost summary statistics."""
        return {
            'total_cost': self.total_slippage + self.total_commission + self.total_market_impact,
            'total_slippage': self.total_slippage,
            'total_commission': self.total_commission,
            'total_market_impact': self.total_market_impact,
            'trade_count': self.trade_count,
            'total_volume_traded': self.total_volume_traded,
            'avg_cost_per_trade': (
                (self.total_slippage + self.total_commission + self.total_market_impact) 
                / self.trade_count
            ) if self.trade_count > 0 else 0,
            'avg_cost_bps': (
                (self.total_slippage + self.total_commission + self.total_market_impact) 
                / self.total_volume_traded * 10000
            ) if self.total_volume_traded > 0 else 0
        }
    
    def get_cost_dataframe(self) -> pd.DataFrame:
        """Get detailed cost history as DataFrame."""
        return pd.DataFrame(self.cost_history)
    
    def reset(self):
        """Reset all tracked costs."""
        self.total_slippage = 0.0
        self.total_commission = 0.0
        self.total_market_impact = 0.0
        self.trade_count = 0
        self.total_volume_traded = 0.0
        self.cost_history = []
