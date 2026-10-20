from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import DEFAULT_CLEAN, NUMERIC_FEATURES, TARGET

# takes in the clean data file created after pre processing, and generate the eda data plots
def generate_eda(data_path: Path = DEFAULT_CLEAN, output_dir: Path = Path("plots/eda")) -> None:
    df = pd.read_csv(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    # class balance plot - depicts if we have enough data for each class, if strong imbalance exists, then accuracy may be misleading
    ax = sns.countplot(data=df, x=TARGET, order=[0, 1])
    ax.set(title="Outcome Balance", xlabel="Class (0: absent, 1: present)", ylabel="Count")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=200)
    plt.close()

    # histogram - distribution and skewness for continous features
    df[NUMERIC_FEATURES].hist(figsize=(12, 8), bins=18, edgecolor="white")
    plt.suptitle("Distributions of Continuous Predictors")
    plt.tight_layout()
    plt.savefig(output_dir / "numerical_histograms.png", dpi=200)
    plt.close()

    # correlation heatmap - inspects relationships, npte-> categorical codes are represented numerically only
    plt.figure(figsize=(12, 9))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".3f", annot_kws={"size": 7})
    plt.title("Pearson Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    missing = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(9, 4))
    sns.barplot(x=missing.index, y=missing.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Missing Feature Values")
    plt.ylabel("Missing records")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=180)
    plt.close()

    summary = {
        "rows": len(df),
        "features": len(df.columns) - 1,
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    df.describe(include="all").to_csv(output_dir / "descriptive_statistics.csv")
    print(f"Created EDA artifacts under {output_dir}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--output", type=Path, default=Path("plots/eda"))
    args = parser.parse_args()
    generate_eda(args.data, args.output)

if __name__ == "__main__":
    main()
