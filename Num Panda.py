import numpy as np
import pandas as pd


def create_random_series(n=10, seed=42):
    rng = np.random.default_rng(seed)
    values = rng.integers(low=1, high=100, size=n)
    return pd.Series(values)


def create_labeled_series(series):
    labels = [f"num_{i + 1}" for i in range(len(series))]
    return pd.Series(series.values, index=labels)


def describe_series(s, title="Series"):
    print(title)
    print(s)

    print("\nDtype :", s.dtype)
    print("Shape :", s.shape)
    print("Size  :", s.size)
    print("Index :", list(s.index))


if __name__ == "__main__":

    # Step 1: Create Series with default index
    series = create_random_series()

    print("STEP 1: Series with the default index")
    describe_series(series)

    # Step 2: Create Series with custom labels
    labeled = create_labeled_series(series)

    print("\nSTEP 2: Same values, custom labels")
    describe_series(labeled, "Labeled Series")

    # Step 3: Basic statistics
    print("\nSTEP 3: Basic statistics")
    print("Minimum :", series.min())
    print("Maximum :", series.max())
    print("Mean    :", series.mean())
    print("Sum     :", series.sum())

    # Step 4: Sorting
    print("\nSTEP 4: Sorted values (ascending)")
    print(series.sort_values())

    # Step 5: Indexing and filtering
    print("\nSTEP 5: Indexing and filtering")
    print("First value :", series.iloc[0])

    print("\nValues greater than 50:")
    print(series[series > 50])