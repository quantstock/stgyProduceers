"""
data query modulus
author: wenping lo
last updated: 2019/8/6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import sys

class Data(object):
    def __init__(self):
        self.__connect_db()

    def get_df_from_db(self, collection_name, stockId, startTime, endTime):
        temp_df = pd.DataFrame(list(self.db[collections].find(
             {"timestamp":{"$gte":startTime, "$lte":endTime},
              "stockId":stockId})))  # selection criterion
        temp_df = temp_df.drop(columns=["_id", "stockId"]).drop_duplicates("timestamp", keep="first").set_index("timestamp")
        temp_df = self.__parse_df_into_float(temp_df)

        return temp_df

    def get_dailyChips(self, stockId, startTime, endTime):
        chip_collections = [
            'dailyCreditTrading',
            'advancedChipsPercentage',
            'dailyFundTrading',
            'dailyDayTrading',
            'dailyOddLots',
            'dailyStockLending']
        temp_dfs = []
        for collections in chip_collections:
            temp_df = pd.DataFrame(list(self.db[collections].find(
                 {"timestamp":{"$gte":startTime, "$lte":endTime},
                  "stockId":stockId})))  # selection criterion
            temp_df = temp_df.drop(columns=["_id", "stockId"]).drop_duplicates("timestamp", keep="first").set_index("timestamp")
            temp_dfs.append(temp_df)
        temp_df = pd.concat(temp_dfs, axis=0)

        return temp_df

    def get_dailyOHLCV(self, stockId, startTime, endTime):
        temp_df = pd.DataFrame(list(self.db["dailyPrice"].find(
                 {"timestamp": {
                     "$gte": startTime, "$lte": endTime},
                     "stockId":stockId},  # selection criterion
                 {"timestamp": "-1",
                  "成交股數": "1",
                  "收盤價": "1",
                  "最低價": "1",
                  "最高價": "1",
                  "開盤價": "1"})))
        # 處理dataframe
        temp_df = temp_df.drop(columns="_id").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        # 重新命名OHLCV
        temp_df = temp_df.rename(columns = {
            "成交股數": "volume",
            "收盤價": "close",
            "最低價": "low",
            "最高價": "high",
            "開盤價": "open"})
        # 將df命名為stockId
        temp_df.name = stockId
        # 將str轉為float
        # temp_df = temp_df.astype(float)
        temp_df["volume"] = temp_df["volume"].apply(lambda x: x.replace(',', '')).astype(float)

        return temp_df

    def get_dailyFundTrading(self, stockId, startTime, endTime, processed=True):
        temp_df = self.get_df_from_db("dailyFundTrading", stockId, startTime, endTime)
        if processed:
            def merge_df_columns(df, target_c, root_c1, root_c2):
                df[target_c] = pd.concat(
                    [df[target_c].dropna(),
                    df[root_c1].dropna().replace(np.nan, 0) +
                    df[root_c2].dropna().replace(np.nan, 0)],
                    axis=0)
                return df
            temp_df = merge_df_columns(temp_df, "自營商賣出股數", '自營商賣出股數自行買賣', '自營商賣出股數避險')
            temp_df = merge_df_columns(temp_df, "自營商買進股數", '自營商買進股數自行買賣', '自營商買進股數避險')
            temp_df = merge_df_columns(temp_df, "外資賣出股數", '外資自營商賣出股數', '外陸資賣出股數不含外資自營商')
            temp_df = merge_df_columns(temp_df, "外資買進股數", '外資自營商買進股數', '外陸資買進股數不含外資自營商')
            temp_df = merge_df_columns(temp_df, "外資買賣超股數", '外資自營商買賣超股數', '外陸資買賣超股數不含外資自營商')
            return temp_df
        else:
            return temp_df

    def __parse_close(self, x):
        try: return pd.to_numeric(x["收盤價"].replace(",", ""))
        except ValueError:
            try: return (pd.to_numeric(x["最後揭示賣價"].replace(",", "")) + pd.to_numeric(x["最後揭示賣價"].replace(",", "")))/2
            except ValueError:
                try: return pd.to_numeric(x["最後揭示賣價"].replace(",", ""))
                except ValueError:
                    try: return pd.to_numeric(x["最後揭示買價"].replace(",", ""))
                    except ValueError:
                        return np.nan

    def __parse_df_into_float(self, df):
        # make all dataframe into float; nan will remain
        def __parse_to_float(x):
            try:
                return float(x.replace(",", ""))
            except:
                return x
        for c in df.columns:
            df[c] = df[c].apply(__parse_to_float)
        return df

    def __connect_db(self):
        mongo_uri = 'mongodb://stockUser:stockUserPwd@localhost:27017/stock_data' # local mongodb address
        dbName = "stock_data" # database name
        self.db = MongoClient(mongo_uri)[dbName]

def plot_columns_time(df):
    #畫出每個columns的時間軸
    df_c = df.columns
    df.columns= [str(s) for s in range(len(df.columns))]
    # create a (constant) Series for each sensor
    for i, sym in enumerate(df.columns):
        t_range = df[[sym]].dropna().index
        dff = t_range.to_series().apply(lambda x: i if x >= t_range.min() and x <= t_range.max() else numpy.NaN)

        p = dff.plot(ylim=[0, len(df.columns)], legend=False)
        p.set_yticks(range(len(df.columns)))
        p.set_yticklabels(df.columns)
    return df_c

if __name__ == '__main__':
    data = Data()
    stockId = "2330"
    startTime = datetime.datetime(2005, 1, 1)
    endTime = datetime.datetime(2019, 8, 6)
    collections = "dailyFundTrading"
    df = data.get_dailyOHLCV(stockId, startTime, endTime)
    df = data.get_dailyFundTrading(stockId, startTime, endTime)
    df = data.get_dailyChips(stockId, startTime, endTime)
    plot_columns_time(df)
