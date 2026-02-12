# -*- coding: utf-8 -*-
import pandas as pd

pool = pd.read_csv('results_full/thematic_stock_pool.csv', encoding='utf-8-sig')
detail = pd.read_csv('results_full/all_stocks_holiday_detail.csv', encoding='utf-8-sig')
summary = pd.read_csv('results_full/holiday_correlation_summary.csv', encoding='utf-8-sig')

sp = detail[detail['holiday']=='春节'].copy()
merged = sp.merge(pool[['ts_code','industry']], on='ts_code', how='left')

keys = ['酒店','旅游','餐饮']
mask_hotel_travel = merged['industry'].fillna('').apply(lambda x: any(k in x for k in keys))
ht = merged[mask_hotel_travel].sort_values('corr_pre7_dummy', ascending=False)
print('酒店/旅游/餐饮 春节样本数:', len(ht))
print(ht[['ts_code','stock_name','industry','corr_pre7_dummy','avg_pre7_cum_return_pct','avg_post7_cum_return_pct','events']].head(20).to_string(index=False))

ht2 = ht[(ht['corr_pre7_dummy']>0) & (ht['events']>=6)]
print('\n酒店/旅游/餐饮 春节 corr>0 且 events>=6:')
print(ht2[['ts_code','stock_name','industry','corr_pre7_dummy','avg_pre7_cum_return_pct','avg_post7_cum_return_pct','events']].head(20).to_string(index=False))

print('\n全样本 best_holiday 的事件数分布:')
print(summary['events'].describe().to_string())
print('events<=4 占比', round((summary['events']<=4).mean(),4))
print('events<=6 占比', round((summary['events']<=6).mean(),4))

print('\n春节样本事件分布:')
print(sp['events'].describe().to_string())
print('春节 events<=4 占比', round((sp['events']<=4).mean(),4))
print('春节 events<=6 占比', round((sp['events']<=6).mean(),4))
