# ai/debug_utils.py
import pandas as pd

def print_integrity_check(df: pd.DataFrame, step_name: str):
    """Prints a detailed integrity report for a DataFrame at a specific step."""
    print("\n" + "="*80)
    print(f"INTEGRITY CHECK: {step_name}")
    print("="*80)

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("DataFrame is EMPTY or not a valid DataFrame.")
        print("="*80 + "\n")
        return

    print(f"Shape: {df.shape}")
    print("\nInfo:")
    df.info()

    nan_counts = df.isna().sum()
    print(f"\nNaN Counts (Features with NaNs only):")
    if nan_counts.sum() == 0:
        print("No NaN values found.")
    else:
        print(nan_counts[nan_counts > 0].sort_values(ascending=False))

    # print("\nHead (first 3 rows):")
    # print(df.head(3))
    print("="*80 + "\n")