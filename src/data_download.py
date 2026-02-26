import yfinance as yf
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
TICKERS = ["SPY", "IEF"]
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

# -----------------------------
# Download Data
# -----------------------------
def download_data():
    print("Downloading data...")

    data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False
    )

    # Keep only Adjusted Close
    adj_close = data["Adj Close"].copy()

    # Drop missing values
    adj_close.dropna(inplace=True)

    print("Download complete.")
    print(f"Dataset shape: {adj_close.shape}")

    return adj_close


# -----------------------------
# Save Data
# -----------------------------
def save_data(df):
    output_path = "data/raw/price_data.csv"
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    df = download_data()
    save_data(df)