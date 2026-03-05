# Machine Learning for Market Regime Prediction and Adaptive Portfolio Allocation
Financial markets exhibit distinct behavioral patterns—periods of calm followed by turbulence—
that traditional static allocation models fail to capture. This project proposes the
development of a machine learning system to predict volatility regimes in financial markets
and construct an adaptive portfolio allocation strategy. Using daily price data for two
ETFs (SPY for equities, IEF for bonds), we define three market regimes based on realized
volatility percentiles. Supervised classification models—Logistic Regression and Random
Forest—are trained on engineered features including returns, volatility measures, and moving
averages. The predicted regimes inform a simple allocation strategy that shifts exposure
toward defensive assets during forecasted high-volatility periods. The strategy is backtested
against static benchmarks (100% equities and 60/40 balanced portfolio) and evaluated using
standard performance metrics. All work is scoped for 70 hours over 5 weeks, delivering a
complete and reproducible machine learning pipeline.
Keywords: Machine Learning, Market Regimes, Volatility Prediction, Portfolio Allocation,
Random Forest, Logistic Regression

