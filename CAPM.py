# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 03:13:32 2021

@author: Situa
"""

# load modules
import numpy as np
import pandas as pd

# risk-free Treasury rate
R_f = 0.0175 / 252

# read in the market data
data = pd.read_csv('capm_market_data.csv')

print(data.head(3))
print(data.tail(3))

#drop data
df = data.copy()
del df['date']
df

returns = df.pct_change(axis=0)
returns.head(5)
returns.dropna(inplace=True)
returns.head(5)

spy = returns.spy_adj_close.values
print(spy[:5])
aapl = returns.aapl_adj_close.values
print(aapl[:5])
type(spy)

aapl_xs = aapl - R_f
spy_xs = spy - R_f

#take a look at first 5 value for aapl
print(aapl_xs[:5])

print(aapl_xs[-5:])
print(spy_xs[-5:])

import matplotlib.pyplot as plt
plt.scatter(spy_xs, aapl_xs)
plt.title("AAPL Excess Returns vs SPY Excess Returns")
plt.xlabel("spy_xs")
plt.ylabel("aapl_xs")
plt.grid()

x = spy_xs.reshape(-1,1)
y = aapl_xs.reshape(-1,1)

xtx = np.matmul(x.transpose(), x)
xtxi = np.linalg.inv(xtx)
xtxixt = np.matmul(xtxi, x.transpose())
beta = np.matmul(xtxixt, y)
beta_hat = np.matmul(xtxixt, y)[0][0] #value at row = 0, column = 0
print('Beta is: ', beta_hat)

#one liner
#beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), x.transpose()), y)[0][0]

def beta_sensitivity(x,y):
    
    out = []
    sz = x.shape[0]
    for ix in range(sz):
        xx = np.delete(x, ix).reshape(-1,1)
        yy = np.delete(y, ix).reshape(-1,1)    
        bi = np.matmul(np.matmul(np.linalg.inv(np.matmul(xx.transpose(), xx)), xx.transpose()), yy)[0][0]
        out.append(tuple([ix,bi]))
    return (out)

betas = beta_sensitivity(x,y)
print(betas[:5])