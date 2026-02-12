# -*- coding: utf-8 -*-
import tushare as ts
import os

ts.set_token(os.environ['TUSHARE_TOKEN'])
pro=ts.pro_api()
df=pro.stock_basic(exchange='',list_status='L',fields='ts_code,name,industry,market')
print('total',len(df))
name_keywords=['酒店','旅游','饮食','餐饮','啤酒','白酒','乳业','饮料','食品','调味','酿酒','零食','卤味','烘焙','景区','免税','民宿']
industry_keywords=['住宿','餐饮','旅游','酒店','酒','饮料','食品','农副食品']
mask_name=df['name'].fillna('').apply(lambda x:any(k in x for k in name_keywords))
mask_ind=df['industry'].fillna('').apply(lambda x:any(k in x for k in industry_keywords))
sel=df[mask_name|mask_ind].copy()
sel=sel[sel['ts_code'].str.endswith('.SH') | sel['ts_code'].str.endswith('.SZ')]
print('selected',len(sel))
print(sel[['ts_code','name','industry']].head(80).to_string(index=False))
print('industry counts')
print(sel['industry'].value_counts().head(30).to_string())
