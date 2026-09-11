"""fons-bench command line. Subcommands are added phase by phase."""
from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fons-bench")
    sub = ap.add_subparsers(dest="cmd")
    bt = sub.add_parser("build-table", help="assemble results/master_v{N}.parquet from data/raw/")
    bt.add_argument("--version", type=int, default=None, help="explicit version number (default: next)")

    ev = sub.add_parser("evaluate", help="score your own predictions with the calibrated thresholds")
    ev.add_argument("--predictions", required=True, help="CSV with columns: system_id, signal[, pocket_similarity, correct]")
    ev.add_argument("--model", required=True, help="which model's calibration to apply")
    ev.add_argument("--alpha", type=float, default=0.1, help="target error rate among accepted (default 0.1)")
    ev.add_argument("--out", default=None, help="write the annotated per-prediction CSV here")

    sub.add_parser("reproduce", help="regenerate every figure from results/master.parquet")
    args = ap.parse_args(argv)
    if args.cmd == "build-table":
        from fons_bench.table import build
        out, meta = build(version=args.version)
        print(f"wrote {out} ({meta['rows']} rows)")
        print(json.dumps(meta, indent=1, default=str))
    elif args.cmd == "evaluate":
        from pathlib import Path

        from fons_bench.evaluate_api import evaluate_predictions, summarise
        df = evaluate_predictions(Path(args.predictions), args.model, args.alpha)
        print(summarise(df, args.alpha).to_string(index=False))
        if "pocket_similarity" not in df.columns:
            print("\nNo `pocket_similarity` column: only the marginal threshold was applied.")
            print("The marginal threshold is NOT valid under distribution shift - that is this")
            print("benchmark's central finding. Supply pocket similarity to get the conditional rule.")
        else:
            uncert = df.loc[~df["bin_certifiable"].fillna(False), "shift_bin"].dropna().unique()
            if len(uncert):
                print(f"\nBins with no threshold achieving alpha={args.alpha} (all predictions rejected): "
                      f"{', '.join(map(str, sorted(uncert)))}")
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\nwrote {args.out}")
    elif args.cmd == "reproduce":
        import runpy
        from pathlib import Path
        script = Path(__file__).resolve().parents[2] / "scripts" / "07_figures.py"
        if not script.exists():
            raise SystemExit("reproduce needs the repository checkout (scripts/07_figures.py)")
        runpy.run_path(str(script), run_name="__main__")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
