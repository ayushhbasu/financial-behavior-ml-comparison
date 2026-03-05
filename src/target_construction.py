import pandas as pd
import numpy as np

# Load Price Data
def load_data(filepath: str = "data/raw/price_data.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    if "SPY" not in df.columns:
        raise ValueError("Expected 'SPY' column not found in data.")
    return df


# Construct Target (Regimes)
def create_target(df: pd.DataFrame) -> pd.Series:
    # Log returns
    returns = np.log(df["SPY"] / df["SPY"].shift(1))

    # 21-day rolling realized volatility (annualised)
    vol_21 = returns.rolling(21).std() * np.sqrt(252)
    vol_21 = vol_21.dropna()

    # Expanding quantile thresholds — no look-ahead bias
    low_thresh  = vol_21.expanding().quantile(0.30)
    high_thresh = vol_21.expanding().quantile(0.70)

    conditions = [vol_21 < low_thresh, vol_21 > high_thresh]
    choices    = [0, 2]

    regimes = pd.Series(
        np.select(conditions, choices, default=1),
        index=vol_21.index,
        name="regime",
        dtype=int,
    )
    return regimes


# Save Target
def save_target(regimes: pd.Series,
                output_path: str = "data/processed/target.csv") -> None:
    regimes.to_csv(output_path, header=True)
    print(f"Target saved → {output_path}")
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    dist = (regimes.value_counts()
                   .sort_index()
                   .rename(index=label_map))
    total = len(regimes)
    for name, cnt in dist.items():
        print(f"  {name:>6}: {cnt:4d}  ({100*cnt/total:.1f}%)")


# Main
if __name__ == "__main__":
    df      = load_data()
    regimes = create_target(df)
    save_target(regimes)
