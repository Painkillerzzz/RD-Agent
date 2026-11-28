"""Does spec B reproduce spec A's signal exactly? Runs under the zoo python (PYTHONPATH=<zoo>/validation).

    python repro_check.py A.py B.py [--seeds 0,1]

Uses a synthetic crypto-perp fixture only (no real data, so nothing about DEV/VAL performance is revealed): 30
symbols, 100 days of hourly bars with a random taker-buy share, 8h funding, hourly OI with long/short and top-trader
ratios. Compares the default params and up to two SPEC variants. NaN and 0 count as the same (both = flat).
Prints one JSON line: {"equal": bool, "max_abs_diff": float, "active_frac": float, "cases": [...]}.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys

import numpy as np
import pandas as pd


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(seed: int, n_sym: int = 30, n_days: int = 100):
    from zoo_val.data import DataBundle
    from zoo_val.fixtures import _gbm_ohlc, _long

    rng = np.random.default_rng(seed)
    syms = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"] + [f"ALT{i:02d}USDT" for i in range(n_sym - 5)]
    hrs = pd.date_range("2022-01-01", periods=n_days * 24, freq="h")
    o, h, l, c, v = _gbm_ohlc(rng, len(hrs), n_sym, 1 / (365 * 24), sigma=0.8)
    b1h = _long(hrs, syms, o, h, l, c, v, crypto=True)
    b1h["taker_buy_volume"] = b1h["volume"] * rng.beta(5, 5, len(b1h))
    b1h["n_trades"] = (b1h["volume"] * rng.lognormal(1, 0.5, len(b1h))).astype(int) + 1
    keep = np.ones(len(b1h), dtype=bool)  # a few late listings -> ragged panel
    for s in syms[-5:]:
        keep &= ~((b1h["symbol"].values == s) & (b1h["ts"].values < np.datetime64(hrs[int(rng.integers(100, 900))])))
    b1h = b1h[keep].reset_index(drop=True)
    days = pd.date_range(hrs[0], hrs[-1], freq="D")
    o, h, l, c, v = _gbm_ohlc(rng, len(days), n_sym, 1 / 365, sigma=0.8)
    b1d = _long(days, syms, o, h, l, c, v, crypto=True)
    fts = pd.date_range(hrs[0], hrs[-1], freq="8h")
    fund = pd.DataFrame({"ts": np.repeat(fts.values, n_sym), "symbol": np.tile(syms, len(fts)),
                         "funding_rate": rng.normal(1e-4, 3e-4, len(fts) * n_sym)})
    oi = pd.DataFrame({"ts": np.repeat(hrs.values, n_sym), "symbol": np.tile(syms, len(hrs))})
    walk = np.exp(np.cumsum(rng.normal(0, 0.01, (len(hrs), n_sym)), axis=0)).ravel()
    oi["open_interest"] = 1e6 * walk
    oi["open_interest_value"] = oi["open_interest"] * 100
    oi["long_short_ratio"] = np.exp(np.cumsum(rng.normal(0, 0.02, (len(hrs), n_sym)), axis=0)).ravel()
    oi["top_trader_ls_ratio"] = np.exp(np.cumsum(rng.normal(0, 0.02, (len(hrs), n_sym)), axis=0)).ravel()
    return DataBundle("crypto-perp", {"bars_1h": b1h, "bars_1d": b1d, "funding": fund, "oi": oi},
                      {"fixture": True, "seed": seed})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args(argv)
    A, B = load(args.a, "spec_a"), load(args.b, "spec_b")
    pa = dict(A.SPEC["params"])
    cases = [("default", {})] + [(v.get("name", f"v{i}"), v.get("params") or {})
                                 for i, v in enumerate((A.SPEC.get("variants") or [])[:2])]
    out = {"equal": True, "max_abs_diff": 0.0, "active_frac": 0.0, "cases": []}
    act = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        bundle = fixture(seed)
        for name, upd in cases:
            p = {**pa, **upd}
            pb = {**dict(B.SPEC["params"]), **upd}
            row = {"seed": seed, "case": name}
            try:
                sa = A.signal(bundle, dict(p))
                sb = B.signal(bundle, dict(pb))
            except Exception as e:  # noqa: BLE001
                row.update(error=f"{type(e).__name__}: {e}"[:600])
                out["equal"] = False
                out["cases"].append(row)
                continue
            sa = sa.astype(float).fillna(0.0)
            sb = sb.astype(float).reindex(index=sa.index, columns=sa.columns).fillna(0.0)
            d = float(np.nanmax(np.abs((sa - sb).to_numpy()))) if sa.size else 0.0
            a_frac = float((sa.to_numpy() != 0).mean()) if sa.size else 0.0
            row.update(max_abs_diff=d, active_frac=round(a_frac, 4), shape=list(sa.shape))
            act.append(a_frac)
            out["max_abs_diff"] = max(out["max_abs_diff"], d)
            if d > 1e-9:
                out["equal"] = False
                diff = (sa - sb).abs()
                r, c = np.unravel_index(int(np.argmax(diff.to_numpy())), diff.shape)
                row["worst"] = {"ts": str(sa.index[r]), "symbol": str(sa.columns[c]),
                                "a": float(sa.iat[r, c]), "b": float(sb.iat[r, c]),
                                "rows_differing": int((diff.to_numpy() > 1e-9).any(axis=1).sum())}
            out["cases"].append(row)
    out["active_frac"] = round(min(act), 4) if act else 0.0
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
