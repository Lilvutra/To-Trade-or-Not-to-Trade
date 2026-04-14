# In this script, we will build the universe of stocks for our analysis
# What do I mean by universe? regime here it is
# Since we need to know which is the current macro regime so that we can focus on the right set of stocks and 
# but how can we know the current macro regime?
# We will use a combination of macroeconomic indicators, market data, and machine learning techniques to identify the current regime

# Diagnose current VN market regime (2026)**
# Identify underexploited data sources that can improve your model**

# 🧠 1. Current Vietnam Market Regime (VERY important)

## Macro regime: **“Growth + Liquidity + Structural distortion”**
# This is a regime where:
# - Growth is strong, driven by public investment and FDI
# - Liquidity is abundant, fueled by retail inflows
# - But market structure is distorted, with narrow leadership and retail dominance
# We will need to create a stock universe that captures this regime effectively
# Regime matrix includes:



### Strong growth + policy support

# GDP target ~6–10% depending on estimate ([The Investor][1])
# Public investment + FDI = key drivers ([SSI][2])
# Earnings growth ~14–20% expected ([Mirae Asset Securities][3])

### ⚠️ But market structure is NOT healthy

### ⚠️ Capital flow divergence

### ⚠️ Macro pressure signals emerging

# USD/VND rising ([Mirae Asset Securities][5])
# Oil spike (war-related) ([Mirae Asset Securities][5])
# Credit tightening risk (real estate sensitive) ([Reuters][6])
# Interest rates stabilizing but still impactful ([vietnam.vn][7])

## 🧠 Regime summary (quant view)

from matplotlib import text


"""
    Regime = 
+ Growth (bullish)
+ Liquidity (bullish)
+ Retail dominance (unstable alpha)
+ Macro shocks (volatile)
+ Sector concentration (inefficient pricing)

"""

# Features
# 1. Domestic vs Foreign Growth
# 2. Credit conditions
# 3. USD/VND trend
# 4. Policy news 

# Build regime model using macro data and market indicators

# macro data = [interest rates, credit growth, usd/vnd, oil prices, vnindex performance, retail inflows, etc] ? 






























































