"""Back-adjust the raw OHLCV cache for inferred splits. Offline; no network.

    python scripts/build_adjusted_cache.py

Reads   data/reference/nifty50_ohlcv_raw.parquet
Writes  data/reference/nifty50_ohlcv.parquet          (adjusted, what NULL loads)
        data/reference/inferred_corporate_actions.csv (every candidate, auditable)

The raw series is kept alongside the adjusted one, so the adjustment is
reproducible and reversible rather than a destructive edit. Re-running this
regenerates the adjusted cache from the raw one; it never edits in place.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "reference" / "nifty50_ohlcv_raw.parquet"
ADJUSTED = REPO / "data" / "reference" / "nifty50_ohlcv.parquet"
EVENTS = REPO / "data" / "reference" / "reconciled_large_moves.csv"


def main() -> int:
    import pandas as pd

    from null.data.adjust import back_adjust, reconcile_moves
    from null.data.corporate_actions import load_corporate_actions

    if not RAW.exists():
        print(f"No raw cache at {RAW}. Run scripts/fetch_bhavcopy.py first.")
        return 2

    frame = pd.read_parquet(RAW)
    frame["date"] = pd.to_datetime(frame["date"])
    symbols = tuple(sorted(frame["symbol"].unique()))
    actions = load_corporate_actions(symbols=symbols)
    events = reconcile_moves(frame, actions)
    accepted = [e for e in events if e.adjusted]
    demergers = [a for a in actions if a.kind == "demerger"]

    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol", "date", "raw_move_pct", "implied_factor",
                "calendar_factor", "adjusted", "residual_move_pct",
                "announced_action",
            ]
        )
        for e in sorted(events, key=lambda x: (x.symbol, x.date)):
            writer.writerow(
                [
                    e.symbol, e.date, f"{e.raw_move * 100:.2f}",
                    f"{e.implied_factor:.4f}", f"{e.calendar_factor:.4f}",
                    e.adjusted, f"{e.residual_move * 100:.2f}", e.calendar_subject,
                ]
            )

    adjusted = back_adjust(frame, actions)
    adjusted.to_parquet(ADJUSTED, index=False)

    print(f"corporate actions loaded : {len(actions)}")
    print(f"large moves reconciled   : {len(events)}")
    print(f"  matched and adjusted   : {len(accepted)}")
    print(f"  no announced action    : {len(events) - len(accepted)}")
    print(f"demergers, never adjusted: {len(demergers)}")
    for d in demergers:
        print(f"    {d.symbol} {d.ex_date}  {d.subject[:56]}")
    print(f"wrote {ADJUSTED}")
    print(f"wrote {EVENTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
