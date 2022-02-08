from dataset import Dataset
import numpy as np
import pandas as pd

import vectorbt as vbt
import pandas_ta as ta
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter(action='ignore')


def create_features(data):
    data['chg'] = data['close'].pct_change(1)
    data['chg_volume'] = data['volume'].pct_change(1)
    data['cuts_price'] = pd.cut(data['chg'], bins=12, labels=False)
    data['cuts_volume'] = pd.cut(data['volume'], bins=12, labels=False)
    candles = data.ta.cdl_pattern(name="all")
    data = pd.concat([data, candles], axis=1)
    data = data.dropna()
    return data, candles


def create_target_names(data):
    target_names = []
    for i in range(1, 13):
        name = f'expected_chg_{i}'
        name_direction = f'expected_dir_{i}'
        target_names.append(name)
        data[name] = data['close'].pct_change(i).shift(-i)
        data[name_direction] = np.where(data[name] > 0, 1, -1)
    return target_names


def generate_statistics(data, signal, patterns, patterns_2, target):
    return data.groupby([patterns, patterns_2])[target]\
        .agg(['sum', 'mean', 'size'])\
        .where(lambda x: (x['mean'] > 0.001) | (x['mean'] < -0.001))\
        .where(lambda x: x['size'] > 30)\
        .query(f'{patterns} == {signal}')\
        .dropna()\
        .reset_index()


def backtest_statistics(data, signal, patterns, target, v):
    if v['mean'] > 0:
        build_signals(data, signal, patterns, patterns_2, target, v)
        return vbt.Portfolio.from_signals(
            close=data['close'],
            entries=data['entries'],
            exits=data['exits'],
            fees=0.001
        )

    else:
        build_signals(data, signal, patterns, patterns_2, target, v)
        return vbt.Portfolio.from_signals(
            close=data['close'],
            short_entries=data['entries'],
            short_exits=data['exits'],
            fees=0.001
        )


def build_signals(data, signal, patterns, patterns_2, target, v):
    target = int(target.split('_')[2])
    rules = (data[patterns] == signal) & \
        (data[patterns_2] == v[patterns_2])
    data['entries'] = np.where(rules, True, False)
    data['exits'] = np.where(rules.shift(target), True, False)


def add_candle_pattetns(
    results, patterns, patterns_2, target, stat, pf_stat, position
):
    trading_statistics = {
        'pf_stat': pf_stat,
        'df_stat': stat,
        'position': position,
        'patterns': patterns,
        'patterns_2': patterns_2
    }
    results[patterns][target] = trading_statistics


def save_plot(signal, patterns, patterns_2, target, v, pf, position):
    pf.plot().write_image(format="png",
                          file=f"backtest_charts/p1-{patterns}_p2-{v[patterns_2]}_t-{target}_p-{position}_s-{signal}.png")


def save_results(results):
    with open('backtest_charts/results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def backtest_strategy(
    data, patterns_2, results, signal, patterns, target, stat, plot=False
):
    for _, v in stat.iterrows():
        data['entries'] = 0
        data['exits'] = 0
        pf = backtest_statistics(data, signal, patterns, target, v)
        pf_stat = pf.stats()
        if pf_stat['Total Return [%]'] < 0:
            continue
        position = 'short' if v['mean'] < 0 else 'long'
        add_candle_pattetns(results, patterns, patterns_2,
                            target, v, pf_stat, position)
        if plot:
            save_plot(signal, patterns, patterns_2,
                      target, v, pf, position)


if __name__ == '__main__':
    data_binance = Dataset().get_data(days=360, ticker='BTCUSDT', ts='1h')
    data = data_binance.copy()
    data, candles = create_features(data)
    target_names = create_target_names(data)
    patterns_2 = 'cuts_price'
    signals = np.unique(data[candles.columns].values)
    results = {}

    for patterns in tqdm(candles.columns):
        if patterns == 'CDL_DOJI_10_0.1':
            continue
        results[patterns] = {}
        for signal in signals:
            if signal == 0:
                continue
            for target in target_names:
                stat = generate_statistics(
                    data, signal, patterns, patterns_2, target)
                if not len(stat):
                    continue

                backtest_strategy(data, patterns_2, results,
                                  signal, patterns, target, stat, plot=False)

    save_results(results)
