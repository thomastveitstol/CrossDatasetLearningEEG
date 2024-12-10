import os.path
import sys

import numpy
import pandas


ROUND = {"mae": 1, "mse": 0, "mape": 2, "pearson_r": 2, "r2_score": 2, "spearman_rho": 2}


def main():
    # L = Lower the better, H = Higher the better
    intercept = True
    scores = {f"results_mae_intercept_{intercept}": "L", f"results_mse_intercept_{intercept}": "L",
              f"results_pearson_r_intercept_{intercept}": "H", f"results_r2_score_intercept_{intercept}": "H"}

    # Get dataframes
    dataframes = []
    for score, direction in scores.items():
        df = pandas.read_csv(f"{os.path.join(os.path.dirname(__file__), score)}.csv")
        dataframes.append(df)

    # Stack the dataframes along a new axis
    stacked = numpy.stack([df.to_numpy() for df in dataframes], axis=0)[..., 2:]

    # Get minimum and maximum
    min_scores = numpy.min(stacked, axis=0)# .fill
    max_scores = numpy.max(stacked, axis=0)

    min_df = pandas.DataFrame(min_scores, index=dataframes[0].index)
    max_df = pandas.DataFrame(max_scores, index=dataframes[0].index)

    # Fixes
    min_df.insert(0, "source_dataset", dataframes[0]["source_dataset"])
    min_df.insert(0, "metric", dataframes[0]["metric"])
    min_df.columns = dataframes[0].columns

    max_df.insert(0, "source_dataset", dataframes[0]["source_dataset"])
    max_df.insert(0, "metric", dataframes[0]["metric"])
    max_df.columns = dataframes[0].columns

    combined_df = min_df.copy()
    for col in min_df.columns:
        if col in ("source_dataset", "metric"):
            continue

        combined_df[col] = [(round(mn, ROUND[metric]), round(mx, ROUND[metric]))
                            for metric, (mn, mx) in zip(combined_df["metric"], zip(min_df[col], max_df[col]))]

    # Print to stdout
    combined_df = combined_df.drop("metric", axis="columns")
    combined_df.to_csv(sys.stdout, sep="&", header=False)


if __name__ == "__main__":
    main()
