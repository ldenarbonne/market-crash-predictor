"""Market Crash Lab - crash-risk early-warning toolkit.

Modules
-------
data       : market data loading (repo snapshot, Yahoo Finance, CSV upload)
synthetic  : regime-switching synthetic market for demos and tests
labels     : forward max-drawdown target and historical crash episodes
features   : backward-looking early-warning indicators
model      : purged walk-forward classification, metrics, alarm evaluation
lppls      : Log-Periodic Power Law Singularity bubble diagnostics
"""

__version__ = "0.1.0"
