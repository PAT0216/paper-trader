"""
Regime Detection Module for Paper Trader - Phase 3.6

Detects market regime based on VIX levels and adjusts trading accordingly.
In elevated/crisis regimes, reduces position sizes to protect capital.

VIX Thresholds (industry standard):
- < 20: Normal volatility (full trading)
- 20-25: Slightly elevated (full trading)
- 25-35: Elevated fear (50% position size)
- > 35: Crisis/panic (no new positions)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class MarketRegime(Enum):
    """Market regime classifications based on VIX."""
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRISIS = "CRISIS"


@dataclass
class RegimeConfig:
    """Configuration for regime thresholds and multipliers."""
    vix_elevated_threshold: float = 25.0
    vix_crisis_threshold: float = 35.0
    normal_multiplier: float = 1.0
    elevated_multiplier: float = 0.5
    crisis_multiplier: float = 0.0  # No new positions


class RegimeDetector:
    """
    Detects market regime and provides position size adjustments.
    
    Usage:
        detector = RegimeDetector()
        regime = detector.get_regime(current_vix)
        multiplier = detector.get_position_multiplier(current_vix)
        adjusted_position = intended_position * multiplier
    """
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
    
    def get_regime(self, vix_value: float) -> MarketRegime:
        """
        Determine market regime from VIX value.
        
        Args:
            vix_value: Current VIX level
            
        Returns:
            MarketRegime enum value
        """
        if vix_value >= self.config.vix_crisis_threshold:
            return MarketRegime.CRISIS
        elif vix_value >= self.config.vix_elevated_threshold:
            return MarketRegime.ELEVATED
        else:
            return MarketRegime.NORMAL
    
    def get_position_multiplier(self, vix_value: float) -> float:
        """
        Get position size multiplier based on VIX.
        
        Args:
            vix_value: Current VIX level
            
        Returns:
            Multiplier (0.0 to 1.0) for position sizing
        """
        regime = self.get_regime(vix_value)
        
        multipliers = {
            MarketRegime.NORMAL: self.config.normal_multiplier,
            MarketRegime.ELEVATED: self.config.elevated_multiplier,
            MarketRegime.CRISIS: self.config.crisis_multiplier,
        }
        
        return multipliers[regime]
    
    def get_regime_info(self, vix_value: float) -> Dict:
        """
        Get complete regime information.
        
        Args:
            vix_value: Current VIX level
            
        Returns:
            Dict with regime, multiplier, and description
        """
        regime = self.get_regime(vix_value)
        multiplier = self.get_position_multiplier(vix_value)
        
        descriptions = {
            MarketRegime.NORMAL: "Normal volatility - full trading enabled",
            MarketRegime.ELEVATED: "Elevated fear - reduced position sizes (50%)",
            MarketRegime.CRISIS: "Crisis mode - no new positions, capital protection",
        }
        
        return {
            'vix': vix_value,
            'regime': regime.value,
            'multiplier': multiplier,
            'description': descriptions[regime],
            'thresholds': {
                'elevated': self.config.vix_elevated_threshold,
                'crisis': self.config.vix_crisis_threshold,
            }
        }
    
    def format_status(self, vix_value: float) -> str:
        """
        Format regime status for logging/output.
        
        Args:
            vix_value: Current VIX level
            
        Returns:
            Formatted status string
        """
        info = self.get_regime_info(vix_value)
        
        emoji = {
            MarketRegime.NORMAL.value: "🟢",
            MarketRegime.ELEVATED.value: "🟡",
            MarketRegime.CRISIS.value: "🔴",
        }
        
        return (
            f"{emoji.get(info['regime'], '⚪')} Regime: {info['regime']} "
            f"(VIX: {vix_value:.2f}, Position: {info['multiplier']*100:.0f}%)"
        )
