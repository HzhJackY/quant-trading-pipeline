"""
Build Shenwan L1 industry mapping from StockClassifyUse_stock.xls.

Output: data/sw_industry.parquet with PIT (point-in-time) industry assignments.
"""
import pandas as pd
import numpy as np

# Standard Shenwan 2014 L1 code -> name mapping
SW_L1_MAP = {
    '11': '农林牧渔', '21': '采掘', '22': '基础化工', '23': '钢铁',
    '24': '有色金属', '25': '建筑材料', '26': '建筑装饰', '27': '电子',
    '28': '汽车', '31': '家用电器', '32': '食品饮料', '33': '家用电器',
    '34': '食品饮料', '35': '纺织服装', '36': '轻工制造', '37': '医药生物',
    '41': '公用事业', '42': '交通运输', '43': '房地产', '44': '商贸零售',
    '45': '建筑装饰', '46': '电力设备', '47': '国防军工', '48': '银行',
    '49': '非银金融', '51': '综合', '61': '传媒', '62': '社会服务',
    '63': '电力设备', '64': '计算机', '65': '国防军工', '71': '通信',
    '72': '传媒', '73': '机械设备', '74': '煤炭', '75': '石油石化',
    '76': '环保', '77': '建筑材料',
}

# Load classification history
df = pd.read_excel('StockClassifyUse_stock.xls')
cols = df.columns.tolist()
df = df.rename(columns={
    cols[0]: 'symbol', cols[1]: 'start_date',
    cols[2]: 'ind_code', cols[3]: 'end_date',
})
df['symbol'] = df['symbol'].astype(str).str.zfill(6)
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
df['l1_code'] = df['ind_code'].astype(str).str[:2]
df['sw_l1'] = df['l1_code'].map(SW_L1_MAP)
df = df.dropna(subset=['sw_l1'])

print(f'Loaded {len(df)} industry assignments for {df["symbol"].nunique()} stocks')
print(f'Date range: {df["start_date"].min()} ~ {df["end_date"].max()}')
print(f'Industry groups: {df["sw_l1"].nunique()}')

# Load CSI800 panel dates
panel = pd.read_parquet('output/training_panel_v3_full.parquet', columns=['date', 'symbol'])
panel['date'] = pd.to_datetime(panel['date'])
panel_syms = sorted(panel['symbol'].unique())
panel_dates = sorted(panel['date'].unique())

# For each (symbol, date), find the industry valid at that date
# Use a simple approach: for each stock, get the effective date range for each industry
result_rows = []
n_matched = 0
n_unmatched = 0

for sym in panel_syms:
    sym_hist = df[df['symbol'] == sym].sort_values('start_date')
    if sym_hist.empty:
        # No industry data -> use '未知'
        for dt in panel_dates:
            result_rows.append({'symbol': sym, 'date': dt, 'sw_l1': '未知'})
        n_unmatched += 1
        continue

    n_matched += 1
    # For each date, find the latest industry assignment with start_date <= date
    for dt in panel_dates:
        valid = sym_hist[sym_hist['start_date'] <= dt]
        if valid.empty:
            sw_l1 = '未知'
        else:
            sw_l1 = valid.iloc[-1]['sw_l1']
        result_rows.append({'symbol': sym, 'date': dt, 'sw_l1': sw_l1})

result = pd.DataFrame(result_rows)

# For sector-relative z-score, we need date-independent industry (one per symbol)
# Use the LATEST assignment for each symbol as the industry
latest_ind = result.sort_values('date').groupby('symbol').tail(1)[['symbol', 'sw_l1']]
latest_ind = latest_ind.drop_duplicates(subset=['symbol'])

print(f'\nCSI 800 coverage: {n_matched}/{len(panel_syms)} stocks with SW L1 data')
print(f'Industry distribution:')
for name, cnt in latest_ind['sw_l1'].value_counts().items():
    print(f'  {name}: {cnt}')

# Save
latest_ind.to_parquet('data/sw_industry.parquet', index=False)
print(f'\nSaved {len(latest_ind)} symbols to data/sw_industry.parquet')
