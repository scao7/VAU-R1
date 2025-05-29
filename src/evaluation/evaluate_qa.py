import argparse
import os 
import csv
import pandas as pd


def main(args):
    gt_df = pd.read_csv(args.test_gt_path)
    pred_df = pd.read_csv(args.pred_path)

    correct_count = 0
    total = len(gt_df)
    for index, row in gt_df.iterrows():
        video_name = str(row["Video Name"])
        correct_option = row["Correct Option"]
        pred_row = pred_df[pred_df["Video Name"] == video_name]
        pred = pred_row["QA answer"].values[0] if not pred_row.empty else []
        correct_count += model_answer.lower() == correct_option.lower()

    final_accuracy = (correct_count / total) * 100

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Total Questions: {total}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
    print(f"Evaluation results saved to {self.save_path}\n", "-"*40, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA results")
    parser.add_argument("--test_gt_path", type=str, required=True, help="Path to the ground truth CSV file")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the result CSV file")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the evaluation results (default: pred_path with _eval suffix)")
    args = parser.parse_args()

    if args.save_path is None:
        save_path = args.pred_path.replace(".csv", "_eval.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    main(args)