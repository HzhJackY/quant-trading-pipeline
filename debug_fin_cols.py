"""Debug: find exact financial data column names."""
import pandas as pd
fin = pd.read_parquet('output/all_financial_pit.parquet')

# Print column index + hex representation
for i, c in enumerate(fin.columns):
    hx = c.encode('utf-8').hex()
    # Known patterns to look for
    n = fin[c].notna().sum()
    print(f'  [{i}] hex={hx[:40]:40s} non-NaN={n}/{len(fin)}')
    if i >= 30:
        break

# Check specific columns by position
print(f'\nColumn 15 (should be operating profit):')
print(f'  name hex: {fin.columns[15].encode("utf-8").hex()}')
print(f'  non-NaN: {fin[fin.columns[15]].notna().sum()}')

print(f'\nColumn 20 (should be equity multiplier):')
print(f'  name hex: {fin.columns[20].encode("utf-8").hex()}')
print(f'  non-NaN: {fin[fin.columns[20]].notna().sum()}')
