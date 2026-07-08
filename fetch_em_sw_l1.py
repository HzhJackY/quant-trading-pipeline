"""
Fetch EM Shenwan Level-1 Industry Boards for 100% CSI 800 Coverage.

EM sub-industry (t:2) classification covers only ~28% of CSI 800 — it misses
large diversified blue chips that don't fit into niche sub-industry boards.

This script identifies the SW L1 boards and fetches their constituents,
then merges with existing sub-industry data for maximum coverage.

Strategy:
  1. Get all EM boards including L1 sectors (m:90+t1 for SW L1 board listing)
  2. For each SW L1 board, fetch all constituents
  3. Merge: SW L1 for 100% coverage, sub-industry for granular within-sector peers
"""
import requests, time, pandas as pd, os
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500,502,503,504], raise_on_status=False)
session.mount('http://', HTTPAdapter(max_retries=retries))
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'http://quote.eastmoney.com/',
}
BASE = 'http://82.push2.eastmoney.com/api/qt/clist/get'
UT = 'bd1d9dd1031a93e1196165d2b4abbd74'

def get_boards(fs_filter, desc):
    """Get board listing."""
    params = {'pn':'1','pz':'500','po':'1','np':'1','ut':UT,
              'fltt':'2','invt':'2','fid':'f3','fs':fs_filter,'fields':'f12,f14,f13'}
    for attempt in range(3):
        try:
            r = session.get(BASE, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200 and r.json().get('data') and r.json()['data'].get('diff'):
                items = r.json()['data']['diff']
                print(f'{desc}: {len(items)} boards')
                return items
        except Exception as e:
            time.sleep(2)
    print(f'{desc}: FAILED')
    return []

# Try different board filters to find SW L1
print('=== Searching for SW L1 boards ===')
all_boards = {}

# Known SW L1 board code ranges
# BK04xx = 申万一级行业
sw_l1_boards = get_boards('m:90+t1', 'm:90+t1 (SW L1)')
all_boards['SW_L1'] = sw_l1_boards

# Also get sub-industry boards
sw_l2_boards = get_boards('m:90+t2+f:!12', 'm:90+t2 (sub-industry)')
all_boards['SW_L2'] = sw_l2_boards

# Print first 15 SW L1 boards
print('\n=== SW L1 Boards ===')
for b in sw_l1_boards[:30]:
    print(f'  {b["f12"]}: {b["f14"]}')

# Now fetch constituents for each SW L1 board
if sw_l1_boards:
    print(f'\n=== Fetching SW L1 constituents ===')
    all_stocks = []
    for idx, b in enumerate(sw_l1_boards):
        b_code = b['f12']
        b_name = b['f14']
        params = {'pn':'1','pz':'2000','po':'1','np':'1','ut':UT,
                  'fltt':'2','invt':'2','fid':'f3',
                  'fs':f'b:{b_code}+f:!50','fields':'f12'}
        try:
            r = session.get(BASE, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                j = r.json()
                if j.get('data') and j['data'].get('diff'):
                    for stock in j['data']['diff']:
                        raw = str(stock['f12'])
                        sym = raw.replace('.SH','').replace('.SZ','').replace('.BJ','').zfill(6)
                        all_stocks.append({'symbol': sym, 'sw_l1': b_name})
            if (idx+1) % 5 == 0:
                print(f'  {idx+1}/{len(sw_l1_boards)} boards, {len(all_stocks)} stocks...')
        except Exception as e:
            print(f'  {b_name}: {e}')
        time.sleep(0.12)

    if all_stocks:
        df = pd.DataFrame(all_stocks)
        df = df.drop_duplicates(subset=['symbol'])

        # Check CSI 800 coverage
        panel = pd.read_parquet('output/training_panel_v3_full.parquet', columns=['symbol'])
        panel_syms = set(panel['symbol'].unique())
        covered = set(df['symbol'].unique())
        overlap = len(covered & panel_syms)

        print(f'\nSW L1 coverage: {len(df)} total stocks')
        print(f'CSI 800 coverage: {overlap}/{len(panel_syms)} ({100*overlap/len(panel_syms):.1f}%)')

        os.makedirs('data', exist_ok=True)
        df.to_parquet('data/sw_industry_l1.parquet', index=False)
        print(f'Saved to data/sw_industry_l1.parquet')
else:
    print('No SW L1 boards found — keeping existing sub-industry data')
