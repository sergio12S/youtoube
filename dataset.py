import pandas as pd
from binance import Client
from dotenv import dotenv_values
from datetime import datetime, timedelta
from typing import List


config = dotenv_values('.env')
client = Client(config.get('KEY'), config.get('SECRET_KEY'))


class Dataset:
    def __init__(self) -> None:
        self.time_frames = [
            '1m',
            '2m',
            '3m',
            '5m',
            '15m',
            '30m',
            '1h',
            '2h',
            '4h',
            '6h',
            '8h',
            '12h',
            '1d',
            '3d',
            '1w',
            '1M'
        ]

    def get_data(self, days: int, ticker: str, ts=str) -> pd.DataFrame():
        return self._download_data(days=days, ticker=ticker, ts=ts)

    def get_datas(self, days: int, tickers: List, ts=str) -> pd.DataFrame():
        datas = [self._download_data(days=days, ticker=i, ts=ts)
                 for i in tickers]
        return pd.concat(datas, axis=0)

    def _download_data(
        self,
        days: int = 10,
        ticker: str = 'BTCUSDT',
        ts: str = '1h'
    ) -> pd.DataFrame():
        if ts not in self.time_frames:
            raise ValueError(f'Support time frames: {self.time_frames}')
        from_day = self._get_start_day(days=days)
        klines = client.get_historical_klines(
            ticker, ts, from_day
        )
        data = self._convert_to_dataframe(klines)
        data['symbol'] = ticker
        return data

    def _convert_to_dataframe(
        self,
        klines: List
    ) -> pd.DataFrame():
        data = pd.DataFrame(
            data=[row[1:7] for row in klines],
            columns=['open', 'high', 'low', 'close', 'volume', 'time'],
        ).set_index('time')
        data.index = pd.to_datetime(data.index + 1, unit='ms')
        data = data.sort_index()
        data = data.apply(pd.to_numeric, axis=1)
        return data

    def _get_start_day(self, days: int) -> str:
        end = datetime.now()
        end = end - timedelta(days=0)
        start = end - timedelta(days=days)
        start = start.strftime('%d %b, %Y')
        return start
