"""Check EM industry coverage against CSI 800 universe."""
import pandas as pd

em = pd.read_parquet('data/sw_industry.parquet')
em_syms = set(em['symbol'].unique())

panel = pd.read_parquet('output/training_panel_v3_full.parquet', columns=['symbol'])
panel_syms = set(panel['symbol'].unique())

overlap = em_syms & panel_syms
em_only = em_syms - panel_syms
panel_only = panel_syms - em_syms

print(f'EM total: {len(em_syms)}')
print(f'CSI 800: {len(panel_syms)}')
print(f'Overlap: {len(overlap)}')
print(f'EM only: {len(em_only)}')
print(f'CSI800 only: {len(panel_only)}')

missing = sorted(list(panel_only))
print(f'\nCSI800 missing from EM ({len(missing)}):')
print(f'First 30: {missing[:30]}')
print(f'Last 10: {missing[-10:]}')
