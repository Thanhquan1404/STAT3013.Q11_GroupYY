# buildpipeline.py

import argparse
from pathlib import Path

from dataset import load_dataset
from train import cross_validate_and_save_folds
from evaluate import evaluate_ensemble
from predict import ensemble_predict_from_csv
from config import TEST_CSV_PATH


# ============================
# Command Functions
# ============================

def cmd_train(_args):
    X, y = load_dataset()
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    cross_validate_and_save_folds(X, y)


def cmd_eval(args):
    test_csv = args.test_csv or TEST_CSV_PATH
    print(f"Evaluating ensemble on: {test_csv}")
    evaluate_ensemble(test_csv)


def cmd_predict(args):
    input_csv = args.input_csv
    output_csv = args.output_csv

    if output_csv is None:
        p = Path(input_csv)
        output_csv = str(p.with_name(p.stem + "_with_preds.csv"))

    print(f"Running predictions on: {input_csv}")
    print(f"Saving output to: {output_csv}")
    df = ensemble_predict_from_csv(input_csv, output_csv)
    print(df.head())


# ============================
# Auto pipeline (no arguments)
# ============================

def run_auto_pipeline():
    print("\n=== AUTO PIPELINE: TRAIN + EVAL ===\n")

    print("Step 1: Training...")
    X, y = load_dataset()
    cross_validate_and_save_folds(X, y)

    print("\nStep 2: Evaluating...")
    evaluate_ensemble(TEST_CSV_PATH)

    print("\n=== AUTO PIPELINE DONE ===")


# ============================
# Argument Parser
# ============================

def build_parser():
    parser = argparse.ArgumentParser(
        description="RandomForest cirrhosis pipeline (train/eval/predict). "
                    "Run without arguments for auto train+eval."
    )

    subparsers = parser.add_subparsers(dest="command")

    # train
    p_train = subparsers.add_parser("train")
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = subparsers.add_parser("eval")
    p_eval.add_argument("--test-csv", type=str, default=None)
    p_eval.set_defaults(func=cmd_eval)

    # predict
    p_pred = subparsers.add_parser("predict")
    p_pred.add_argument("input_csv", type=str)
    p_pred.add_argument("--output-csv", type=str, default=None)
    p_pred.set_defaults(func=cmd_predict)

    return parser


# ============================
# Entry Point
# ============================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        # no args -> run training + evaluate
        run_auto_pipeline()
    else:
        # run requested subcommand
        args.func(args)


if __name__ == "__main__":
    main()

# # 1) Train and save 6 fold models
# python buildpipeline.py train

# # 2) Evaluate on default TEST_CSV_PATH from config.py
# python buildpipeline.py eval

# #    Or specify your own test file
# python buildpipeline.py eval --test-csv D:\PTTK\some_test.csv

# # 3) Predict on a new file and auto-save <name>_with_preds.csv
# python buildpipeline.py predict D:\PTTK\new_patients.csv

# #    Or custom output path
# python buildpipeline.py predict D:\PTTK\new_patients.csv --output-csv D:\PTTK\new_with_preds.csv

# # 4) Run full pipeline (train + eval) with no arguments
# python buildpipeline.py