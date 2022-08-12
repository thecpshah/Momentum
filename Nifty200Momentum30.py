import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import time

# Read Nift200 list
df = pd.read_csv('ind_nifty200list.csv')

# Drop unnecessary columns
df.drop(["Series", "ISIN Code", "Industry"], axis = 1, inplace = True)

# Create Yahoo symbol
df['Yahoo_Symbol'] = df.Symbol + '.NS'

symbol = df['Yahoo_Symbol'].tolist()

# Dates
startDate = "2021-08-12"
endDate = "2022-08-12"

# Momentum Ratio 6 and 12
MR6 = []
MR12 = []

for k in symbol:
	print(k)
	try:
		# Get data for symbol
		dfSymbol = pdr.get_data_yahoo(k, startDate, endDate)

		# Create date, Month and Year column
		dfSymbol["Date"] = dfSymbol.index
		dfSymbol["Month"] = dfSymbol["Date"].dt.month 
		dfSymbol["Year"] = dfSymbol["Date"].dt.year 
		grps = dfSymbol.groupby(["Year", "Month"])

		# Calculate daily and monthly returns
		dailyReturns = dfSymbol["Adj Close"].pct_change()
		monthlyReturns = dfSymbol["Adj Close"].resample('M').ffill().pct_change()

		# days for volatility. (252?)
		days = dfSymbol.shape[0] - 1

		monthlyClose = pd.DataFrame()

		for k in grps:
			monthlyClose = monthlyClose.append(k[1].tail(1), ignore_index = True)

		# Price return 6 months
		priceReturn6 = monthlyClose["Adj Close"][12]/monthlyClose["Adj Close"][6] - 1

		voltality6 = np.std(np.log1p(dailyReturns[1:])) * np.sqrt(days)

		# Momentum ratio 6 months for the symbol
		momentumRatio6 = priceReturn6 / voltality6
		MR6.append(momentumRatio6)

		# Price return 12 months
		priceReturn12 = monthlyClose["Adj Close"][12]/monthlyClose["Adj Close"][0] - 1

		voltality12 = np.std(np.log1p(dailyReturns[1:])) * np.sqrt(days)

		# Momentum ratio 12 months for the symbol
		momentumRatio12 = priceReturn12 / voltality12
		MR12.append(momentumRatio12)

		time.sleep(0.25)
	except Exception:
		print("----------Symbol not found")
		pass

df["MR6"] = MR6
df["MR12"] = MR12

# Z Score for 6 and 12 months
df["ZScore6"] = (MR6 - np.mean(MR6)) / np.std(MR6)
df["ZScore12"] = (MR12 - np.mean(MR12)) / np.std(MR12)

# Weighted Average Z Score
df["WAvZScore"] = 0.5*df["ZScore6"] + 0.5*df["ZScore12"]

# Normalised momentum score
df["NormMomentumScore"] = 0

df.loc[df["WAvZScore"] >= 0, "NormMomentumScore"] = df[df["WAvZScore"] >= 0]["WAvZScore"] + 1

df.loc[df["WAvZScore"] < 0, "NormMomentumScore"] = 1 / (1 - df[df["WAvZScore"] < 0]["WAvZScore"])

df.sort_values("NormMomentumScore", ascending = False, inplace=True)
df.reset_index(drop=True, inplace=True)
top30 = df[:30]