import pandas as pd
import mplfinance as mpf
import numpy as np
import talib as ta
import time

from backtesting import Backtest, Strategy
from backtesting.lib import crossover, barssince
from backtesting.test import SMA, GOOG

from pathlib import Path
current_path = Path.cwd()

#Raw Data Handle
# df = pd.read_csv(current_path / 'Data/1Day_only_trading_time.csv',delimiter=',')

# df['Date'] = df['Date'] + ' ' + df['Time']
# df = df.drop(columns=['Time'])
# df.set_index('Date',inplace=True)
# df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M:%S")

# df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H:%M:%S")
# df = df.set_index('Date')

# df = df.resample('D').agg({'Open': 'first', 
#                                 'High': 'max', 
#                                 'Low': 'min', 
#                                 'Close': 'last',
#                                 'Volume': 'sum'}).dropna()

# df['Open'] = round(df.rolling(window=60)['Open'].std())
# df['High'] = round(df.rolling(window=60)['High'].std())
# df['Low'] = round(df.rolling(window=60)['Low'].std())
# df['Close'] = round(df.rolling(window=60)['Close'].std())

# df.to_csv('1Day_only_trading_time_std_60.csv')

# print(df)
# if None:
#     print('hihi')
# time.sleep(60)

print('--------------------------')    
the_std =14 

df = pd.read_csv(current_path / 'Data/3min_only_trading_time.csv',delimiter=',')

df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
df = df.set_index('Date')

# df = df.iloc[:150000]

# # ###! to drop not used dataframe
# x = len(df)
# pp = []
# for i in range(0,len(df)):
#     # print(df.index[i])
#     # print(int(df.index[i].hour))
#     if df.index[i].dayofweek in [5,6]:
#         print(df.index[i])
#         pp.append(i)
#         x-=1
#     if df.index[i].hour in [16,17,18,19,20,21,22,23,0,1,2,3,4,5,6,7,8]:
#         if int(df.index[i].hour) in [16] and int(df.index[i].minute) in [0]:
#             pass
#         else:
#             pp.append(i)
#             x-=1

#     if x == i + 1:
#         break
# df = df.drop(df.index[pp])
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df)
# # ###! to drop not used dataframe

# # df = df.asfreq('3min', method='bfill')
# df.to_csv('3min_only_trading_time.csv')

# df['MA'] = round(df.rolling(window=21)['Close'].mean())
# df['std'] = round(df.rolling(window=3000)['Close'].std())

agg_dict = {'Close': 'last'}
df_close_daily = df.resample('D').agg(agg_dict).dropna()
print(df_close_daily.head())
print(df_close_daily.tail())

df_close_daily['Std'] = round(df_close_daily.rolling(window=the_std)['Close'].std())
print(df_close_daily.head())
print(df_close_daily.tail())

# df_close = df.resample('D').agg(agg_dict).dropna()
# print(type(df_close_daily), type(df_close))
# df_close_daily = pd.merge(df_close, df_close_daily, how='left',on=["Close"])
df_close_daily.to_csv('std_daily.csv')
print('done')
# time.sleep(50)

df_close_daily = df_close_daily.asfreq('3min', method='ffill')
# df = pd.concat([df, df_close_daily], axis=1)
# df_std_daily = round(df_close_daily.rolling(window=10)['Close'].std())
# df['std'] = df_close_daily

print(df_close_daily.head())
print(df_close_daily.tail())

df['std'] = df_close_daily['Std'].fillna(method='ffill')
print(df.head())
print(df.tail())

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df)

df = df.fillna(method='ffill')
print(df)
# time.sleep(50)
# higher_than = True
# previous_higher_than = True
# over_night = True
# range_high = 0
# previous_plot_time = 0

# for i in range(100, len(df.index)):
#     ###Playing With MA
#     if df.iloc[i]['Close'] - df.iloc[i]['MA'] < 0:
#         higher_than = False
#     else:
#         higher_than = True

#     if higher_than and not previous_higher_than:
#         BREAKTHOUGH_MA = True
#     else:
#         BREAKTHOUGH_MA = False
#     if not higher_than and previous_higher_than:
#         BREAKDOWN_MA = True
#     else:
#         BREAKDOWN_MA = False

#     if BREAKTHOUGH_MA or BREAKDOWN_MA:
#         range_high = df.iloc[i-50:i]['Close'].max()
#     previous_higher_than = higher_than    

#     if df.iloc[i]['Close'] > range_high:
#         plot_time = i
#         if df.index[i].hour - df.index[i - 10].hour < 2:
#             over_night = False
#         if plot_time - previous_plot_time > 50:
#             if df.index[i].hour not in [9,15]:
#                 df_show = df.iloc[i-100:i+50]
#                 signal = []
#                 for x in range(0, len(df_show)):
#                     # print(df_show.iloc[x])
#                     if df_show.iloc[x].name == df.iloc[i].name:
#                         signal.append(df.iloc[i]['Close'] * 0.999)
#                     else:
#                         signal.append(np.nan)
#                 ap = [mpf.make_addplot(signal, scatter=True, markersize=200, marker='^', color='b')]
#                 if not over_night:
#                     mpf.plot(df_show, type='candle', mav=(21), addplot=ap, volume=True, title=str(df.iloc[i].name))
#         previous_plot_time = plot_time
# mpf.plot(df, type='candle', mav=(21))
# print(df)

# df = df.iloc[300000:]

class SmaCross(Strategy):
    n1 = 1000
    n2 = 7000
    n_sigma = 3000
    n = 7000
    SL_RATIO = 0.00205
    DAY_TRADE = True
    MARKET = ['HK']
    NUMBEROFSIGMA = 2

    def init(self):
        self.nowtime = self.data.index

        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2) 
        # self.sigma = self.I(ta.STDDEV, self.data.Close, self.n_sigma) 
        self.sigma = self.data.std
        self.open = 0

        self.uptrend = False

    def now2(self, array):
        return pd.Index(array)

    def next(self):
        # if self.equity:
        #     initial_equity = 150000
        # if crossover(self.sma1, self.sma2):
        #     self.uptrend = True
        # elif crossover(self.sma2, self.sma1):
        #     self.uptrend = False   


        #? open chase
        if (self.data.index[self.n].hour == 9 and self.data.index[self.n].minute in range(32,35)):
            # if self.uptrend:
            if self.data.Close[self.n - 5: self.n - 1].max() < self.data.Close[self.n]:
                if int(self.data.std[self.n - 110]) < min(int(self.data.std[self.n - 220]), int(self.data.std[self.n - 330]),  int(self.data.std[self.n - 440]),  int(self.data.std[self.n - 550]),  int(self.data.std[self.n - 660])) :
                    # print([self.n - 110])
                # if int(self.data.std[self.n]) > 1000:
                # print(self.n, self.data.Close[self.n])
                    if not self.position:
                        self.buy(size=50, sl=self.data.Close[self.n] - self.data.Close[self.n] * self.SL_RATIO)#, tp=self.data.Close[self.n] + 200) #50)
                    # if self.equity:
                    #     if self.equity - initial_equity > 150000:
                    #         initial_equity +=150000
                    #         self.buy(size=50, sl=self.data.Close[self.n] - self.data.Close[self.n] * self.SL_RATIO)#, tp=self.data.Close[self.n] + 200) #50)

            # if not self.uptrend:
            if self.data.Close[self.n - 5: self.n - 1].min() > self.data.Close[self.n]:
                if int(self.data.std[self.n - 110]) < min(int(self.data.std[self.n - 220]), int(self.data.std[self.n - 330]),  int(self.data.std[self.n - 440]),  int(self.data.std[self.n - 550]),  int(self.data.std[self.n - 660])) :
                # if int(self.data.std[self.n]) > 1000:
                # print(self.n, self.data.Close[self.n])
                    if not self.position:
                        self.sell(size=50, sl=self.data.Close[self.n] + self.data.Close[self.n] * self.SL_RATIO)#, tp=self.data.Close[self.n] - 200) #50)
                    # if self.equity:
                    #     if self.equity - initial_equity > 150000:
                    #         initial_equity +=150000
                    #         self.sell(size=50, sl=self.data.Close[self.n] + self.data.Close[self.n] * self.SL_RATIO)#, tp=self.data.Close[self.n] - 200) #50)

        # #? Sigma
        # if self.data.index[self.n].hour == 9 and self.data.index[self.n].minute == 30 :      
        #     self.open = self.data.Close[self.n] 
        # if (self.data.index[self.n].hour in [9] and self.data.index[self.n].minute in range(30,59)) or (self.data.index[self.n].hour in range(10,15) and self.data.index[self.n].minute in range(0,59)):
        #         if self.trades or self.position or self.orders:
        #             pass
        #         else:
        #             today_had_trade = False
        #             if self.closed_trades:
        #                 for x in self.closed_trades:
        #                     if x.exit_time.day == self.data.index[self.n].day:
        #                         today_had_trade = True
        #             if not today_had_trade:                    
        #                 if self.data.Close[self.n] < (self.open + (self.NUMBEROFSIGMA * self.sigma)):# * (1 - self.SL_RATIO):
        #                     # if self.uptrend:
        #                     self.sell(size=50, sl=self.data.Close[self.n] * (1 + 2 * self.SL_RATIO)) #50)
        #                 if self.data.Close[self.n] > (self.open - (self.NUMBEROFSIGMA * self.sigma)):# * (1 + self.SL_RATIO):
        #                     # if not self.uptrend:
        #                     self.buy(size=50, sl=self.data.Close[self.n] * (1 - 2 * self.SL_RATIO)) #50)

        if self.trades:
            if self.trades[0].is_long:
                if self.trades[0].entry_price < self.data.Close[self.n] * (1 - self.SL_RATIO): #50):
                    for x in self.trades:
                        x.sl = self.data.Close[self.n] * (1 - self.SL_RATIO) #50)

            if self.trades[0].is_short:
                if self.trades[0].entry_price > self.data.Close[self.n] * (1 + self.SL_RATIO): #50):
                    for x in self.trades:
                        x.sl = self.data.Close[self.n] * (1 + self.SL_RATIO) #50)
            if self.DAY_TRADE and 'HK' in self.MARKET:
                if self.data.index[self.n].hour == 15 and self.data.index[self.n].minute == 57 :
                    for x in self.trades:
                        x.close()
                    # print(self.trades)
        self.n+=1                

bt = Backtest(df, SmaCross,
            cash=160000, commission=.0001, margin=0.001)

output = bt.run()
bt.plot()
print(output)