def feature_price_volatility_20d(df):
    """20-day rolling volatility of returns"""
    returns = df['ISPR_PX_LAST'].pct_change()
    volatility = returns.rolling(window=20).std()
    return volatility.to_frame('price_volatility_20d')

def feature_price_momentum_10d(df):
    """10-day price momentum (return over 10 days)"""
    momentum = df['ISPR_PX_LAST'].pct_change(10)
    return momentum.to_frame('price_momentum_10d')

def feature_volume_spike_ratio(df):
    """Volume spike ratio: current volume vs 20-day average"""
    vol_ma = df['ISPR_PX_VOLUME'].rolling(window=20).mean()
    volume_ratio = df['ISPR_PX_VOLUME'] / vol_ma
    return volume_ratio.to_frame('volume_spike_ratio')

def feature_bid_ask_spread(df):
    """Bid-ask spread as percentage of mid price"""
    mid_price = (df['ISPR_PX_BID'] + df['ISPR_PX_ASK']) / 2
    spread = (df['ISPR_PX_ASK'] - df['ISPR_PX_BID']) / mid_price
    return spread.to_frame('bid_ask_spread_pct')

def feature_put_call_ratio(df):
    """Put to call open interest ratio"""
    put_call_ratio = df['ISPR_OPEN_INT_TOTAL_PUT'] / (df['ISPR_OPEN_INT_TOTAL_CALL'] + 1e-8)
    return put_call_ratio.to_frame('put_call_ratio')

def feature_price_rsi_14d(df):
    """14-day RSI indicator"""
    delta = df['ISPR_PX_LAST'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi.to_frame('price_rsi_14d')

def feature_volume_price_trend(df):
    """Volume-weighted price trend over 10 days"""
    vwap = (df['ISPR_PX_LAST'] * df['ISPR_PX_VOLUME']).rolling(window=10).sum() / df['ISPR_PX_VOLUME'].rolling(window=10).sum()
    price_trend = (df['ISPR_PX_LAST'] - vwap) / vwap
    return price_trend.to_frame('volume_price_trend')

def feature_price_acceleration(df):
    """Price acceleration: change in momentum over 5 days"""
    returns = df['ISPR_PX_LAST'].pct_change()
    momentum = returns.rolling(window=5).mean()
    acceleration = momentum.diff(5)
    return acceleration.to_frame('price_acceleration')

def feature_options_activity_ratio(df):
    """Total options activity relative to stock volume"""
    total_options = df['ISPR_OPEN_INT_TOTAL_PUT'] + df['ISPR_OPEN_INT_TOTAL_CALL']
    options_volume_ratio = total_options / (df['ISPR_PX_VOLUME'].rolling(window=5).mean() + 1e-8)
    return options_volume_ratio.to_frame('options_activity_ratio')

def feature_price_distance_from_high(df):
    """Distance from 30-day high as potential resistance level"""
    rolling_high = df['ISPR_PX_LAST'].rolling(window=30).max()
    distance_from_high = (df['ISPR_PX_LAST'] - rolling_high) / rolling_high
    return distance_from_high.to_frame('price_distance_from_high')