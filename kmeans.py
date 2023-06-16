import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.cluster import KMeans
import numpy as np

# Load historical Bitcoin price data into a Pandas DataFrame
exchange = ccxt.binance()
ohlcvs = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=72000)
df = pd.DataFrame(ohlcvs, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
df.set_index('Date', inplace=True)
df.drop(columns=['Timestamp'], inplace=True)

# Define the number of clusters to use for KMeans algorithm
n_clusters = 5

# Initialize the KMeans clustering algorithm with the specified number of clusters
kmeans = KMeans(n_clusters=n_clusters)

# Fit the algorithm to the data
kmeans.fit(df[['High', 'Low']].values)

# Get the cluster centers and sort them by their y-coordinate
centers = sorted(kmeans.cluster_centers_, key=lambda x: x[1])

print(centers)

# Plot the candlestick chart
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title('BTC/USDT Candlestick Chart')
ax.set_ylabel('Price ($USDT)')
ax.grid(True)

# Create the candlestick chart using mpf.plot()
mpf.plot(df, type='candle', ax=ax)

# Draw the support and resistance levels
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'white', 'orange', 'gray']
for i, center in enumerate(centers):
    ax.axhline(y=center[0], linestyle='--', linewidth=1, color=colors[i])
    ax.text(df.index[0], center[1], 'Resistance ' + str(i+1), color=colors[i], fontsize=12, fontweight='bold')
    
for i, center in enumerate(reversed(centers)):
    ax.axhline(y=center[1], linestyle='--', linewidth=1, color=colors[n_clusters-1-i])
    ax.text(df.index[0], center[1], 'Support ' + str(i+1), color=colors[n_clusters-1-i], fontsize=12, fontweight='bold')
plt.show()

# calculate the average of each bidimensional array
result_array = np.array([np.mean(array) for array in centers])
print(result_array)

df['Levels'] = pd.Series(dtype=float)
df['Levels'] = 0.0
for center in result_array:
    for index, row in df.iterrows():
        if center >= df.loc[index,'Low'] and center <= df.loc[index, 'High']:
            df.loc[index, 'Levels'] = 1.0


df.to_csv("btc_levels.csv")