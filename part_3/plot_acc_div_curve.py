from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def plotting_accdiv_curve(
    results: pd.DataFrame,
    output_path: Path,
    split: str
):
    """ Plot the accuracy-diversity frontier"""

    sweep = pd.read_csv("part_3/validation_metrics.csv").sort_values("lambda")


    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax = sweep.plot.line(
        x="ILD@10",
        y="NDCG@10",
        marker="o",
        label="MMR sweep",
        title="Validation accuracy–diversity curve",
    )

    baseline = sweep.loc[sweep["lambda"].eq(1.0)]
    ax.scatter(
        baseline["ILD@10"],
        baseline["NDCG@10"],
        marker="*",
        s=160,
        color="tab:orange",
        zorder=3,
        label="Markov baseline (λ = 1)",
    )
    ax.set(
        title=f"MMR accuracy–diversity curve ({split})",
        xlabel="ILD@10",
        ylabel="NDCG@10",
    )
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.show()