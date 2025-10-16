"""
RCMAS (Region Control Multi-Agent Systems) Simulator

A comprehensive framework for simulating and analyzing competitive region control
scenarios with multiple agents.
"""

__version__ = "0.1.0"
__author__ = "Steven Jordaan, Nils Timm"

from domain.agent import Agent
from domain.territory import Territory, Sector
from domain.state import State
from domain.action import Action, ActionProfile
from simulation.simulator import RCMASSimulator

__all__ = [
    "Agent",
    "Territory",
    "Sector",
    "State",
    "Action",
    "ActionProfile",
    "RCMASSimulator",
]
