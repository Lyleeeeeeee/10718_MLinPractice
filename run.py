"""Command entry point for the final WikiFlow comparison."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    stages = {"assemble": "data_pipeline.assemble", "acquire": "data_pipeline.acquisition", "prepare": "data_pipeline.preparation",
              "text-plan": "textual.preparation", "text": "textual.acquisition", "encode": "textual.encoding"}
    if len(sys.argv) > 1 and sys.argv[1] in stages:
        import importlib
        importlib.import_module(stages[sys.argv[1]]).main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("show", "verify", "verify-data", "refit", "test"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data", type=Path, help="Prepared data directory; default bundled inputs")
    parser.add_argument("--compare-reference", action="store_true", help="Compare a source-data refit against bundled final scores")
    args = parser.parse_args()
    if args.command == "show":
        with (ROOT / "results/test_summary.csv").open() as stream:
            for row in csv.DictReader(stream):
                print(f"{row['model_id']:26} hits={float(row['hits_at_50']):.3f}/{row['events']} recall={float(row['pooled_recall_at_50']):.4f}")
    elif args.command == "verify":
        from data_analysis.verify import verify
        print(json.dumps(verify(ROOT), indent=2))
    elif args.command == "verify-data":
        if args.data is None:
            parser.error("verify-data requires --data")
        from data_analysis.verify import audit_data
        _, _, _, count = audit_data(ROOT, args.data)
        print(json.dumps(dict(source_rows_checked=count, source_contracts_passed=True)))
    elif args.command == "refit":
        if args.output is None:
            parser.error("refit requires --output runs/<new-directory>")
        from data_analysis.pipeline import refit
        print(json.dumps(refit(args.output, ROOT, args.data, True if args.compare_reference else None), indent=2))
    else:
        subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
