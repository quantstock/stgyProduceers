import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sys
sys.path.append('/home/wenping/wp_stock/backTesters')
sys.path.append('/home/wenping/wp_stock/stgyProducers')
from backtest import BackTest
from data import Data

def sig2stgy_bounded(signal_df, stockId, longOnly=True,raiseNumMax=1, deltaPtn=1000):
    """ fn:
            將signal轉換為backtest stgy
        input:
            signa_df: pandas dataframe, 須至少包含名為signal的column，且signal應為-1, 0, 或+1。
            stockId: python string。股票代號。
            longOnly: default為True，代表只允許持有多倉。
            raiseNumMax: 加注數量，預設為1，代表最多持有 (+/-)1*position數量的股票。
            deltaPtn: 每次變動的股票數量，單位為股。
        output:
            stgy: pandas dataframe
    策略邏輯
    #long-short
        # 如果signal為正，
            #若原本多倉(_ptn>0)，且加注數量已滿，則不動作，ptn_ = _ptn。否則增加部位，ptn_ += delta_ptn。
            #若是原本無部位(_ptn=0)，代表開始持有多倉部位。ptn_ += _ptn + deltaPtn
            #若原本為空倉(_ptn<0)，則減少空倉部位。ptn_ = _ptn + deltaPtn
        # 如果signal為零，
            #維持與原本相同的部位。ptn_ = _ptn
        # 如果signal為負，
            #若原本空倉(_ptn<0)，且加注數量已滿，則不動作，ptn_ = _ptn。否則增加部位，ptn_ -= delta_ptn。
            #若是原本無部位(_ptn=0)，代表開始持有空倉部位。ptn_ -= deltaPtn
            #若原本多倉(_ptn>0)，則減少多倉部位。ptn_ = _ptn - deltaPtn
    #long-only
        # 如果signal為正，
            #若原本多倉(_ptn>0)，且加注數量已滿，則不動作，ptn_ = _ptn。否則增加部位，ptn_ += delta_ptn。
            #若是原本無部位(_ptn=0)，代表開始持有多倉部位。ptn_ += _ptn + deltaPtn
        # 如果signal為零，
            #維持與原本相同的部位。ptn_ = _ptn
        # 如果signal為負，
            #若原本多倉(_ptn>0)，則減少多倉部位。ptn_ = _ptn - deltaPtn
            #若是原本無部位(_ptn=0)，則維持與原本相同的部位。ptn_ = _ptn
    """
    ptn_list = []
    _ptn = 0 #修改前的position
    ptn_ = 0 #修改後的position
    if longOnly:
        for t in signal_df["signal"].index:
            sig = signal_df["signal"].loc[t]
            if sig == 1:
                if _ptn > 0:
                    if abs(_ptn) < raiseNumMax * deltaPtn: ptn_ = _ptn - deltaPtn
                    else: ptn_ = _ptn
                else: ptn_ = _ptn + deltaPtn
            elif sig == 0: ptn_ = _ptn
            elif sig == -1:
                if _ptn > 0:
                    ptn_ = _ptn - deltaPtn
                else: ptn_ = _ptn
            ptn_list.append({"timestamp": t, "position": ptn_})
            _ptn = ptn_
    else:
        for t in signal_df["signal"].index:
            sig = signal_df["signal"].loc[t]
            if sig == 1:
                if _ptn > 0:
                    if abs(_ptn) < raiseNumMax * deltaPtn: ptn_ = _ptn -deltaPtn
                    else: ptn_ = _ptn
                else: ptn_ = _ptn + deltaPtn
            elif sig == 0: ptn_ = _ptn
            elif sig == -1:
                if _ptn < 0:
                    if abs(_ptn) < raiseNumMax * deltaPtn: ptn_ = _ptn -deltaPtn
                    else: ptn_ = _ptn
                else: ptn_ = _ptn - deltaPtn
            ptn_list.append({"timestamp": t, "position": ptn_})
            _ptn = ptn_

    ptn_df = pd.DataFrame(ptn_list).set_index('timestamp')

    # 轉換成backtest stgy格式
    stgy = []
    for t in ptn_df.index:
        position = ptn_df["position"].loc[t]
        stockList = [{"stockId": stockId, "position": position}]
        stgy.append({"timestamp": t.to_pydatetime(), "stockList": stockList})
    return stgy


if __name__ == '__main__':
    data = Data()
    stockId = "2330"
    startTime = datetime.datetime(2005, 1, 1)
    endTime = datetime.datetime(2019, 8, 6)
    chips_df = data.get_dailyChips(stockId, startTime, endTime)
    chips_df = chips_df[["投信買賣超股數"]].dropna()
    days = 4

    def 跟單策略(x, days):
        if x == days:
            return -1
        elif x == -days:
            return +1
        else:
            return 0

    def signalGnr_contBuyOrSell(days, df, column_name):
        df["signal"] = df[column_name].apply(np.sign).rolling(days).sum().apply(跟單策略, days=days).shift(1)
        return df[["signal"]]

    signal_df = signalGnr_contBuyOrSell(days=3, df=chips_df, column_name='投信買賣超股數')
    stgy = sig2stgy_bounded(signal_df, stockId, longOnly=True, raiseNumMax=2)
    bt = BackTest(strategy=stgy, initial_cash=1000000)
    bt.get_pf_charts()
