from PIL import Image
import os
from matplotlib import pyplot as plt
import mplfinance as mpf
import pandas as pd

from pathlib import Path
current_path = Path.cwd()

def search_directory(directory,name):
    name = name.lower()  # Convert up front in case it's pass mixed case
    for root, dirs, files in os.walk(directory,topdown=True):
        for e in files + dirs:
            if os.path.splitext(e)[0].lower() == name:
                yield os.path.join(root, e)

df = pd.read_csv(current_path / 'Data/1Day.csv')
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors='coerce')
df = df.set_index(df['Date'])
del df['Date']


start_date = '2010-5-30'
end_date = '2010-6-30'
df = df.loc((df['Open'].index[0] > start_date) & (df['Open'].index[0] <= end_date))
print(df)

img, axlist = mpf.plot(df, type='candle', returnfig=True)
img.savefig('stock.png', bbox_inches='tight')

file_name = 'stock'
search_path = r'local\Futures'

img = Image.open(next(search_directory(search_path, file_name))).convert('LA')
img.save(r'/Data/output/stock_greyscale.png')

