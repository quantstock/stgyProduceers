import datetime
from pymongo import MongoClient
import pandas as pd
import numpy as np
from functools import reduce


class stgyProducer(object):
    def __init__(self, stockIds, probas, preds, tss, dayAvg=1, potn=1000, odd=False, short=False):
        self.probas = probas
        self.preds = preds
        self.stockIds = stockIds
        self.tss = tss
        self.potn = potn
        self.odd = odd
        self.dayAvg = dayAvg
        self.short = short

    def get_stgy(self):
        self.stgy = self.__get_stgy()
        return self.stgy

    def __get_stgy(self):
        stgys = []; acc=True;
        for iS in range(len(self.stockIds)):
            timestamps = self.tss[iS]

            proba_df = pd.DataFrame(self.probas[iS], index=timestamps, columns=["prob_-1", "prob_+1"])
            pred_df = pd.DataFrame(self.preds[iS], index=timestamps, columns=["label"])
            bt_df = pd.concat([proba_df, pred_df], axis=1)
            bt_df_week = bt_df.resample("{}D".format(self.dayAvg)).mean()

            count = 0; stgy = []
            for t in bt_df_week.index:
                if count == 0:
                    if bt_df_week.loc[t]["label"] > 0:
                        stgy.append({'timestamp': t, 'position': self.potn})
                        count += 1
                    elif bt_df_week.loc[t]["label"] < 0:
                        if self.short:
                            pos = -self.potn
                        else:
                            pos = 0
                        stgy.append({'timestamp': t, 'position': pos})
                        count += 1
                    else:
                        continue
                elif count > 0:
                    if bt_df_week.loc[t]["label"] > 0:
                        stgy.append({'timestamp': t, 'position': stgy[count-1]['position']+self.potn})
                        count += 1
                    elif bt_df_week.loc[t]["label"] < 0:
                        pos = stgy[count-1]['position']-self.potn
                        if self.short:
                            stgy.append({'timestamp': t, 'position': pos})
                        else:
                            if pos > 0:
                                stgy.append({'timestamp': t, 'position': pos})
                            else:
                                stgy.append({'timestamp': t, 'position': 0})
                        count += 1
                    else:
                        continue

            df=pd.DataFrame(stgy).set_index('timestamp')
            self.df = df
            df.columns = pd.MultiIndex.from_tuples([(self.stockIds[iS], 'position')])

            stgys.append(df)

        if len(stgys) < 1:
            return stgys[0]
        else:
            return reduce(self.__pd_merge, stgys)

    def __pd_merge(self, sa, sb):
        return pd.merge(sa, sb, right_index=True, left_index=True)
