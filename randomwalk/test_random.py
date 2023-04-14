import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import vectorbt as vbt


def generate_random_walk_std(
    n: int = 100,
    std: int = 2
) -> np.ndarray:
    """
    Generate a random walk of length n.
    """
    x = np.random.randn(n)
    y = np.cumsum(x)
    start_with = y.min()
    return abs(start_with) + 10 + y + std * np.std(y)


def plot_random_walk(
    n: int = 100,
    std: int = 2
) -> None:
    """
    Plot a random walk of length n.
    """
    y = generate_random_walk_std(n, std)
    plt.plot(y)
    plt.show()


def generate_random_walk_mean(
    n: int = 100,
    mean: int = 0
) -> np.ndarray:
    """
    Generate a random walk of length n.
    """
    x = np.random.randn(n)
    y = np.cumsum(x)
    return y + mean


def plot_random_walk_mean(
    n: int = 100,
    mean: int = 0
) -> None:
    """
    Plot a random walk of length n.
    """
    y = generate_random_walk_mean(n, mean)
    plt.plot(y)
    plt.show()


def generate_random_walk_mean_std(
    n: int = 100,
    mean: int = 0,
    std: int = 2
) -> np.ndarray:
    """
    Generate a random walk of length n.
    """
    x = np.random.randn(n)
    y = np.cumsum(x)
    return y + mean + std * np.std(y)


def plot_random_walk_mean_std(
    n: int = 100,
    mean: int = 0,
    std: int = 2
) -> None:
    """
    Plot a random walk of length n.
    """
    y = generate_random_walk_mean_std(n, mean, std)
    plt.plot(y)
    plt.show()


for _ in range(10):
    plot_random_walk(100, 2)

# attach plots to one figure
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i, j in itertools.product(range(10), range(10)):
    axs[i, j].plot(generate_random_walk_std(100, 1))
    axs[i, j].set_title('std=1')
plt.show()


def simulate_strategy(
    strategies: int = 1000,
    sigma: int = 1
    ) -> tuple:

    data_statistics = []
    data_returns = []
    SIGMA = 1
    for _ in range(strategies):
        # create mean reversing trading srategy
        data = pd.DataFrame(
            {'close': generate_random_walk_std(1000, 1)}
        )
        data['chg'] = data['close'].pct_change()
        std_dev = data['chg'].std()

        data['resistance'] = 0 + std_dev * SIGMA
        data['support'] = 0 - std_dev * SIGMA
        # data[['chg', 'resistance', 'support']].plot()
        data['signal'] = 0
        data['signal'] = np.where(data['chg'] > data['resistance'], 1, 0)
        data['signal'] = np.where(
            data['chg'] < data['support'], -1, data['signal'])

        data['entries'] = np.where(data['signal'] == 1, True, False)
        data['exits'] = np.where(data['signal'] == -1, True, False)
        data['short_entries'] = np.where(data['signal'] == -1, True, False)
        data['short_exits'] = np.where(data['signal'] == 1, True, False)

        pf = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=data['entries'],
            exits=data['exits']
        )
        returns = pf.total_return()
        series_returns = pf.returns()
        # print(f'Strategy {strategy} returns: {returns}')
        data_returns.append(returns)
        statistics = {
            'mean': np.mean(series_returns),
            'std': np.std(series_returns),
            'min': np.min(series_returns),
            'max': np.max(series_returns),
            'positive': len([i for i in series_returns if i > 0]) / len(series_returns),
        }
        data_statistics.append(statistics)
    print(np.mean(data_returns))

    # plot histogram of data_statistics
    data_statistics = pd.DataFrame(data_statistics)
    data_returns = pd.DataFrame(data_returns)
    return data_returns, data_statistics


btc_price = vbt.BinanceData.download(
    ['BTCUSDT'], start='1200 hours ago', end='now UTC', interval='5m')
data = btc_price.get('Close')
data = data.rename('close')
data['chg'] = data['close'].pct_change()
data = pd.DataFrame(data)
