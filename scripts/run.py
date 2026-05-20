"""Command line entry point for the volatility targeting strategy.

Examples
--------
Run with everything from config.yaml::

    python scripts/run.py

Target a higher risk level with the EWMA estimator::

    python scripts/run.py --target-vol 0.15 --method ewma

Save the charts::

    python scripts/run.py --save-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vol_targeting.data import YFinanceLoader, daily_returns
from vol_targeting.strategy import StrategyConfig, run_strategy
from vol_targeting.metrics import format_summary, summarize, annualized_volatility
from vol_targeting import plotting


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the volatility targeting backtest.")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
    )
    p.add_argument("--asset", help="Override the ticker.")
    p.add_argument("--start", help="Override the start date (YYYY-MM-DD).")
    p.add_argument("--end", help="Override the end date (YYYY-MM-DD).")
    p.add_argument("--target-vol", type=float, help="Target annual volatility.")
    p.add_argument("--method", choices=["rolling", "ewma"], help="Vol estimator.")
    p.add_argument("--window", type=int, help="Rolling window in trading days.")
    p.add_argument("--max-leverage", type=float, help="Cap on exposure.")
    p.add_argument("--cost-bps", type=float, help="Transaction cost in bps.")
    p.add_argument("--no-cache", action="store_true", help="Force a fresh download.")
    p.add_argument("--save-plots", action="store_true", help="Write charts to results/.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    asset = args.asset or cfg["asset"]
    start = args.start or cfg["period"]["start"]
    end = args.end or cfg["period"]["end"]

    strat_cfg = StrategyConfig(
        target_annual_vol=args.target_vol or cfg["target"]["annual_volatility"],
        method=args.method or cfg["estimator"]["method"],
        window=args.window or cfg["estimator"]["window"],
        ewma_lambda=cfg["estimator"]["ewma_lambda"],
        max_leverage=args.max_leverage or cfg["position"]["max_leverage"],
        transaction_cost_bps=(
            args.cost_bps
            if args.cost_bps is not None
            else cfg["costs"]["transaction_cost_bps"]
        ),
        risk_free_rate=cfg.get("risk_free_rate", 0.0),
    )

    print(f"Loading {asset} from {start} to {end} ...")
    loader = YFinanceLoader(use_cache=not args.no_cache)
    prices = loader.load(asset, start, end)
    rets = daily_returns(prices)

    print("Running strategy ...")
    result = run_strategy(rets, strat_cfg)

    stats = summarize(result.returns, risk_free_rate=strat_cfg.risk_free_rate)
    bh_stats = summarize(result.buy_and_hold, risk_free_rate=strat_cfg.risk_free_rate)

    print(f"\nVolatility targeted ({strat_cfg.target_annual_vol:.0%} target)")
    print("-" * 42)
    print(format_summary(stats))
    print("\nBuy and hold")
    print("-" * 42)
    print(format_summary(bh_stats))

    realized = annualized_volatility(result.returns)
    print(
        f"\nTarget vol {strat_cfg.target_annual_vol:.2%} | "
        f"strategy realized {realized:.2%} | "
        f"buy-and-hold realized {annualized_volatility(result.buy_and_hold):.2%}"
    )

    if args.save_plots:
        f1 = plotting.plot_equity_curve(result.equity_curve, result.buy_and_hold)
        f2 = plotting.plot_leverage(result.leverage)
        f3 = plotting.plot_realized_vol(result.realized_vol, strat_cfg.target_annual_vol)
        out1 = plotting.save_figure(f1, "results/equity_curve.png")
        out2 = plotting.save_figure(f2, "results/leverage.png")
        out3 = plotting.save_figure(f3, "results/realized_vol.png")
        print(f"\nSaved charts to {out1}, {out2}, {out3}")


if __name__ == "__main__":
    main()
