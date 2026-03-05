import pandas as pd
import numpy as np

# -----------------------------
#  Data
# -----------------------------
def load_data(path: str = "data/raw/price_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if "SPY" not in df.columns:
        raise ValueError("Expected 'SPY' column not found.")
    return df

# -----------------------------
# Feature Engineering
# -----------------------------pyt
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    returns  = df.pct_change()

    # Momentum / return signals
    features["r1"]  = returns["SPY"]
    features["r5"]  = df["SPY"].pct_change(5)
    features["r21"] = df["SPY"].pct_change(21)

    # Volatility
    features["vol_21"] = (
        returns["SPY"]
        .rolling(window=21)
        .std()
        * np.sqrt(252)
    )

    # Trend: distance from moving averages
    ma50  = df["SPY"].rolling(50).mean()
    ma200 = df["SPY"].rolling(200).mean()
    features["ma50_dist"]  = (df["SPY"] - ma50)  / ma50
    features["ma200_dist"] = (df["SPY"] - ma200) / ma200

    # Cross-asset spread (risk-on / risk-off signal)
    features["spy_ief_spread"] = returns["SPY"] - returns["IEF"]

    features.dropna(inplace=True)
    return features

# -----------------------------
# Saving the Features
# -----------------------------
def save_features(features: pd.DataFrame,
                  path: str = "data/processed/features.csv") -> None:
    features.to_csv(path)
    print(f"Features saved → {path}")
    print(f"  Shape   : {features.shape}")
    print(f"  Columns : {features.columns.tolist()}")

if __name__ == "__main__":
    df       = load_data()
    features = create_features(df)
    save_features(features)
