#!/usr/bin/env python3
"""Single entry point: show models, replay saved results, verify, or explicitly refit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from wikiflow.registry import MODELS, resolve_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("models", help="Show retained models and historical IDs")
    listing.add_argument("--model")
    replay = commands.add_parser("replay", aliases=["reproduce"], help="Recompute saved results with zero model fits")
    replay.add_argument("--output", type=Path)
    commands.add_parser("verify", help="Verify inputs, saved metrics and frozen selection")
    commands.add_parser("test", help="Run offline unit and reproduction tests")
    fit = commands.add_parser("fit", help="Explicitly refit one existing development recipe")
    fit.add_argument("--model", required=True)
    fit.add_argument("--month", required=True)
    fit.add_argument("--output", type=Path)
    fit.add_argument("--allow-fit", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "models":
            print(json.dumps(resolve_model(args.model) if args.model else MODELS, indent=2))
        elif args.command == "test":
            environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
            return subprocess.call([sys.executable, "-B", "-m", "unittest", "discover",
                                    "-s", "tests", "-v"], cwd=ROOT, env=environment)
        elif args.command == "fit":
            if not args.allow_fit:
                parser.error("Fitting requires --allow-fit; replay performs 0 fits")
            from wikiflow.models import EARLY_MONTHS
            from wikiflow.core import FORECASTS
            if args.month not in EARLY_MONTHS + FORECASTS:
                parser.error("Only January 2024--August 2026 development folds are allowed")
            if args.output is None or args.output.exists():
                parser.error("Fitting requires a fresh --output file")
            if any(args.output.resolve().is_relative_to(ROOT / name)
                   for name in ("src", "data", "results", "tests", ".git")):
                parser.error("Output cannot be inside source or frozen-input directories")
            from wikiflow.data import load, save
            from wikiflow.workflow import load_study, verify_inputs
            from wikiflow.models import fit_model
            verify_inputs()
            spec = resolve_model(args.model)
            panel, _, _ = load_study()
            documents = {int(key): value for key, value in load(ROOT / "data/historical_text.json.gz").items()}
            scores, info = fit_model(spec["id"], args.month, panel, documents)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save(args.output, dict(model_id=spec["id"], legacy_id=spec["legacy_id"],
                                  month=args.month, predictions=scores, verification=info))
            print(f"Saved {spec['name']}, {args.month}; {info['model_fits']} fixed-recipe fits")
        else:
            from wikiflow.workflow import replay
            main_table, ablation, verification = replay(getattr(args, "output", None))
            if args.command != "verify":
                for title, rows in [("Main results", main_table), ("Text availability ablation", ablation)]:
                    print(title)
                    print(f"{'Model':42} {'Hits':>7} {'Recall':>8} {'NDCG':>8} {'Precision':>9}")
                    for row in rows:
                        ndcg = f"{row['ndcg_at_50']:.4f}" if row["ndcg_at_50"] is not None else "—"
                        print(f"{row['model']:42} {row['hits']:7.2f} {row['recall_at_50']:8.4f} {ndcg:>8} {row['precision_at_50']:9.4f}")
            print(f"PASS: {verification['scored_models']} saved models, {verification['method_months']} method-months, "
                  f"{verification['selection_records_verified']} selection records; 0 fits; no holdout access")
        return 0
    except (ValueError, AssertionError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
