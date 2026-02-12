# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('results_full/all_stocks_holiday_detail.csv', encoding='utf-8-sig')
print('holidays:', sorted(df['holiday'].dropna().unique().tolist()))
sp = df[df['holiday'] == '春节'].copy()
sp = sp.sort_values('corr_pre7_dummy', ascending=False)

print('春节样本股票数:', len(sp))
print('\n=== 春节 corr_pre7_dummy Top20 ===')
print(sp[['ts_code','stock_name','corr_pre7_dummy','avg_pre7_cum_return_pct','avg_post7_cum_return_pct','events']].head(20).to_string(index=False))

print('\n=== 春节 corr>0 且 events>=6 Top20 ===')
sp2 = sp[(sp['corr_pre7_dummy'] > 0) & (sp['events'] >= 6)]
print(sp2[['ts_code','stock_name','corr_pre7_dummy','avg_pre7_cum_return_pct','avg_post7_cum_return_pct','events']].head(20).to_string(index=False))

print('\n=== 春节 按avg_pre7_cum_return_pct Top20 ===')
sp3 = sp.sort_values('avg_pre7_cum_return_pct', ascending=False)
print(sp3[['ts_code','stock_name','corr_pre7_dummy','avg_pre7_cum_return_pct','avg_post7_cum_return_pct','events']].head(20).to_string(index=False))

print('\n春节点总体统计:')
print('mean corr', round(sp['corr_pre7_dummy'].mean(), 6))
print('median corr', round(sp['corr_pre7_dummy'].median(), 6))
print('corr>0 占比', round((sp['corr_pre7_dummy'] > 0).mean(), 4))
print(sp['events'].describe().to_string())
