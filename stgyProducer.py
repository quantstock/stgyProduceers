from datetime import datetime, timedelta
from pymongo import MongoClient
import pandas as pd
import numpy as np
from functools import reduce


class stgyProducer(object):
    def __init__(self, stockIds, probas, preds, tss, dayAvg=1, potn=1000, odd=False, price=False):
        self.db = self.__connect_db()
        self.probas = probas
        self.preds = preds
        self.stockIds = stockIds
        self.tss = tss
        self.potn = potn
        self.odd = odd
        self.dayAvg = dayAvg

    def get_stgy(self):
        self.stgy = self.__get_stgy()
        return self.stgy

    def __get_stgy(self):
        N = 2800; stgys = []; n = 100; acc=False;
        for iS in range(len(self.stockIds)):
            timestamps = self.tss[iS][N+n:]

            proba_df = pd.DataFrame(self.probas[iS], index=timestamps, columns=["prob_-1", "prob_+1"])
            pred_df = pd.DataFrame(self.preds[iS], index=timestamps, columns=["label"])
            bt_df = pd.concat([proba_df, pred_df], axis=1)
            # bt_df_week = bt_df.resample("{}D".format(self.dayAvg)).sum()
            bt_df_week = bt_df.resample("{}D".format(self.dayAvg)).mean()
            # bt_df_week = bt_df

            count = 0; stgy = []
            for t in bt_df_week.index:
                # if bt_df_week.loc[t]["label"] > dayAvg-3:
                if bt_df_week.loc[t]["label"] > -1:
                    # print(count)
                    if count > 0 and acc:
                        if stgy[count-1]['stockList'][0]['position'] >= 0:
                            stockList = [{"stockId":self.stockIds[iS], "position": stgy[count-1]['stockList'][0]['position']+self.potn}]
                    else:
                        if self.odd:
                            if bt_df_week.loc[t]["label"] > 0:
                                stockList = [{"stockId":self.stockIds[iS], "position": self.potn*(2*bt_df_week.loc[t]["prob_+1"]-1)}]
                                # stockList = [{"stockId":self.stockIds[iS], "position": 0}]
                            else:
                                stockList = [{"stockId":self.stockIds[iS], "position": self.potn}]
                        else:
                            stockList = [{"stockId":self.stockIds[iS], "position": self.potn}]
                # elif bt_df_week.loc[t]["label"] < -(dayAvg-3):
                elif bt_df_week.loc[t]["label"] < 1:
                    stockList = [{"stockId":self.stockIds[iS], "position": 0}]
                else:
                    continue
                stgy.append({"timestamp": t.to_pydatetime(), "stockList": stockList})
                count += 1

            stgys.append(stgy)

        # 合成投資組合

        stgysList=[]

        # 利用pd.merge 先制作大stgyDf
        for i in stgys:
            stgysList.append(pd.DataFrame(i).set_index('timestamp'))

        stgysList = reduce(self.__redMerge, stgysList)
        stgysList.columns = self.stockIds


        # 將nan轉成空list
        for col in stgysList.columns:
            for row in stgysList.loc[stgysList[col].isnull(), col].index:
                stgysList.at[row, col] = []

        # 整合至第一行
        for col in self.stockIds[1:]:
            stgysList[self.stockIds[0]]+=stgysList[col]

        # 取出
        t4=stgysList[self.stockIds[0]].reset_index()

        # 製作策略組合
        newStgy=[]
        for i in range(len(t4)):
            newStgy.append({'timestamp': t4.iloc[i]['timestamp'], 'stockList':t4.iloc[i][self.stockIds[0]]})

        return newStgy

    def __redMerge(self, t1, t2):
        return t1.merge(t2, how="outer", left_index=True, right_index=True)

    def __connect_db(self):
        mongo_uri = 'mongodb://stockUser:stockUserPwd@localhost:27017/stock_data' # local mongodb address
        dbName = "stock_data" # database name
        return MongoClient(mongo_uri)[dbName]
