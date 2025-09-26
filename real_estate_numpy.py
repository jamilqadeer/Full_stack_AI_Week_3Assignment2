# pandas_realestate_full.py
# Put RealEstate-USA.csv in same folder, or set CSV to a raw GitHub URL.

import pandas as pd
import numpy as np

CSV = "RealEstate-USA.csv"   # <-- default: local file in same folder
CSV = "https://raw.githubusercontent.com/ShahzadSarwar10/Fullstack-WITH-AI-B-3-SAT-SUN-6Months-Explorer/refs/heads/main/DataSetForPractice/RealEstate-USA.csv"  # optional raw URL

def find_col(df, want):
    """Find column name in df ignoring case/whitespace; returns actual column name or None."""
    want_norm = want.strip().lower()
    for c in df.columns:
        if c.strip().lower() == want_norm:
            return c
    for c in df.columns:
        if want_norm in c.strip().lower():
            return c
    return None

def safe_read(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV: {csv_path}")
        return df
    except Exception as e:
        raise SystemExit(f"Failed to load CSV '{csv_path}': {e}\nMake sure file exists or paste the raw GitHub URL into CSV variable.")

def show_title(t):
    print("\n" + "="*80)
    print(t)
    print("="*80 + "\n")

def analyze_and_print(msg, obj):
    print(f"--- {msg} ---")
    print(obj)
    print()

def main():
    df = safe_read(CSV)

    # Task 1: Load into DataFrame with default indexing and print DataFrame
    show_title("Task 1: Full DataFrame (first 20 rows shown)")
    print("Default auto-indexed DataFrame preview (head 20):")
    print(df.head(20).to_string(index=True))
    print(f"\nFull columns: {list(df.columns)}")

    # identify required columns robustly
    cols_req = {}
    for req in ["brokered_by","price","acre_lot","city","house_size","street","zip_code","state"]:
        cols_req[req] = find_col(df, req)

    # Print found column mapping
    show_title("Column name mapping (requested -> found)")
    for k, v in cols_req.items():
        print(f"{k}  ->  {v}")
    print()

    # Coerce numeric columns to numeric (safe)
    for n in ["price","house_size","acre_lot"]:
        col = cols_req.get(n)
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Task 2: call .info(), .dtypes, .describe(), .shape and analyze
    show_title("Task 2: DataFrame methods/properties")
    print("df.info() output:")
    df.info()
    print("\n.dtypes:")
    print(df.dtypes)
    print("\n.describe() (numeric columns):")
    print(df.describe(include=[np.number]).to_string())
    print("\n.shape:")
    print(df.shape)
    print("\nAnalysis (short):")
    print("- .info() shows rows, non-null counts and memory usage.")
    print("- .dtypes shows types (object, int64, float64).")
    print("- .describe() gives count/mean/std/min/max/percentiles for numeric columns.")
    print("- .shape is (rows, columns).")

    # Task 3: explore DataFrame.to_string() with example parameters
    show_title("Task 3: DataFrame.to_string() examples")
    sample_df = df.head(12)
    print("Default to_string():")
    print(sample_df.to_string())
    print("\nWith columns subset and no index:")
    cols_subset = [c for c in [cols_req.get("city"), cols_req.get("price"), cols_req.get("house_size")] if c]
    print(sample_df.to_string(columns=cols_subset, index=False))
    print("\nWith na_rep and float_format, limited width:")
    print(sample_df.to_string(na_rep="(missing)", float_format=lambda x: f"{x:,.2f}" if pd.notna(x) else x, line_width=80))
    print("\nWith header aliases and col_space:")
    header_alias = ["CityName" if c==cols_req.get("city") else c for c in sample_df.columns]
    # header parameter can accept list of names:
    try:
        print(sample_df.to_string(header=header_alias, col_space=15))
    except Exception as e:
        print("Header alias example not supported in this pandas version:", e)
    print("\nWith max_rows / max_cols (truncated):")
    print(sample_df.to_string(max_rows=5, max_cols=3))
    print("\n(show_dimensions True):")
    print(sample_df.to_string(show_dimensions=True))

    # Task 4: top 7 rows
    show_title("Task 4: Top 7 rows (head(7))")
    print(df.head(7).to_string())

    # Task 5: bottom 9 rows
    show_title("Task 5: Bottom 9 rows (tail(9))")
    print(df.tail(9).to_string())

    # Task 6: access 'city' column and 'street' column (print whole column)
    show_title("Task 6: Access city and street columns")
    city_col = cols_req.get("city")
    street_col = cols_req.get("street")
    if city_col:
        print(f"Column '{city_col}' (whole column, first 20 shown):")
        print(df[city_col].head(20).to_string(index=True))
    else:
        print("City column not found.")
    if street_col:
        print(f"\nColumn '{street_col}' (whole column, first 20 shown):")
        print(df[street_col].head(20).to_string(index=True))
    else:
        print("Street column not found (no problem if dataset uses different naming).")

    # Task 7: access multiple columns like street and city
    show_title("Task 7: Select multiple columns (street, city)")
    multi_cols = [c for c in [street_col, city_col] if c]
    if multi_cols:
        print(df.loc[:, multi_cols].head(10).to_string())
    else:
        print("Required columns not found to show both street & city.")

    # Task 8: Selecting single row using .loc with index 5
    show_title("Task 8: .loc[index=5]")
    if 5 in df.index:
        print(df.loc[5].to_string())
        print("Analysis: .loc uses label-based indexing; since default index = 0..n-1, label 5 is the 6th row.")
    else:
        print("Index label 5 not present (DataFrame index range may differ).")

    # Task 9: Selecting multiple rows using .loc with indices 3,5,7
    show_title("Task 9: .loc[[3,5,7]]")
    wanted = [i for i in [3,5,7] if i in df.index]
    if wanted:
        print(df.loc[wanted].to_string())
    else:
        print("None of indices 3,5,7 found in index.")

    # Task 10: Slice rows using .loc range 3 to 9
    show_title("Task 10: .loc[3:9] (inclusive)")
    # loc slice includes both endpoints
    available_start = 3 in df.index
    # loc with labels will return any labels between 3 and 9 — if they exist
    try:
        slice_loc = df.loc[3:9]
        print(slice_loc.to_string())
        print("Analysis: .loc uses labels and is inclusive of the end label.")
    except Exception as e:
        print("Error using .loc slice:", e)

    # Task 11: Conditional selection price > 100000
    show_title("Task 11: price > 100000")
    price_col = cols_req.get("price")
    if price_col:
        res = df.loc[df[price_col] > 100000]
        print(res.to_string())
        print(f"Analysis: {len(res)} rows where price > 100000.")
    else:
        print("Price column not found.")

    # Task 12: city == 'Adjuntas'
    show_title("Task 12: city == 'Adjuntas'")
    if city_col:
        res = df.loc[df[city_col] == "Adjuntas"]
        print(res.to_string() if not res.empty else "No rows with city == 'Adjuntas'.")
        print("Analysis: check case and spelling; use .str.lower() if you want case-insensitive matching.")
    else:
        print("City column not found.")

    # Task 13: city == 'Adjuntas' and price < 180500
    show_title("Task 13: city == 'Adjuntas' AND price < 180500")
    if city_col and price_col:
        res = df.loc[(df[city_col] == "Adjuntas") & (df[price_col] < 180500)]
        print(res.to_string() if not res.empty else "No rows match both conditions.")
    else:
        print("City or Price column missing.")

    # Task 14: Selecting single row using .loc index 7 and columns specified
    show_title("Task 14: .loc[7, ['city','price','street','zip_code','acre_lot']]")
    cols_interest = [cols_req.get(x) for x in ["city","price","street","zip_code","acre_lot"] if cols_req.get(x)]
    if 7 in df.index and cols_interest:
        try:
            print(df.loc[7, cols_interest])
        except Exception as e:
            print("Error selecting row/cols:", e)
    else:
        print("Index 7 or requested columns not available.")

    # Task 15: Selecting slice of columns using .loc from 'city' to 'zip_code'
    show_title("Task 15: Select columns from 'city' to 'zip_code' (df.loc[:, 'city':'zip_code'])")
    if city_col and cols_req.get("zip_code"):
        try:
            print(df.loc[:, city_col:cols_req.get("zip_code")].head(10).to_string())
        except Exception as e:
            print("Error selecting column slice:", e)
    else:
        print("Required column names for this slice not found.")

    # Task 16: Combined row and column selection using .loc for city 'Adjuntas' and columns city:zip_code
    show_title("Task 16: .loc[df.city=='Adjuntas', 'city':'zip_code']")
    if city_col and cols_req.get("zip_code"):
        res = df.loc[df[city_col] == "Adjuntas", city_col:cols_req.get("zip_code")]
        print(res.to_string() if not res.empty else "No rows for city 'Adjuntas' or columns missing.")
    else:
        print("Required columns missing for this combined selection.")

    # Task 17: Selecting single row using .iloc select 5th row (iloc[4])
    show_title("Task 17: .iloc (5th row) -> iloc[4]")
    try:
        row5 = df.iloc[4]
        print(row5.to_string())
        print("Analysis: .iloc uses 0-based integer positions. 5th row = iloc[4].")
    except Exception as e:
        print("Cannot select 5th row via iloc (index out of range):", e)

    # Task 18: .iloc select 7th, 9th, 15th rows -> positions 6,8,14
    show_title("Task 18: .iloc rows 7th,9th,15th (positions 6,8,14)")
    idxs = []
    for pos in [6,8,14]:
        if pos < len(df):
            idxs.append(pos)
    if idxs:
        print(df.iloc[idxs].to_string())
    else:
        print("One or more requested iloc positions out of range.")

    # Task 19: .iloc slice from 5th to 13th row -> iloc[4:13]
    show_title("Task 19: .iloc slice 5th to 13th -> iloc[4:13]")
    try:
        print(df.iloc[4:13].to_string())
    except Exception as e:
        print("Error:", e)

    # Task 20: .iloc select 3rd column -> iloc[:,2]
    show_title("Task 20: .iloc select 3rd column (position 2)")
    try:
        col3 = df.iloc[:, 2]
        print(col3.head(15).to_string())
    except Exception as e:
        print("Cannot select 3rd column:", e)

    # Task 21: .iloc select multiple columns 2nd, 4th, 7th -> positions 1,3,6
    show_title("Task 21: .iloc select columns positions [1,3,6]")
    cols_pos = [1,3,6]
    cols_existing = [p for p in cols_pos if p < df.shape[1]]
    if cols_existing:
        try:
            print(df.iloc[:, cols_existing].head(10).to_string())
        except Exception as e:
            print("Error selecting multiple columns:", e)
    else:
        print("Requested column positions out of range.")

    # Task 22: .iloc slice columns 2nd to 5th -> iloc[:,1:5]
    show_title("Task 22: .iloc columns slice 2nd to 5th -> iloc[:,1:5]")
    try:
        print(df.iloc[:, 1:5].head(10).to_string())
    except Exception as e:
        print("Error:", e)

    # Task 23: Combined .iloc rows [7,9,15] and columns [2nd,4th] -> positions rows [6,8,14], cols [1,3]
    show_title("Task 23: Combined .iloc rows [6,8,14] and cols [1,3]")
    rows_pos = [p for p in [6,8,14] if p < len(df)]
    cols_pos = [p for p in [1,3] if p < df.shape[1]]
    if rows_pos and cols_pos:
        try:
            print(df.iloc[rows_pos, cols_pos].to_string())
        except Exception as e:
            print("Error:", e)
    else:
        print("Requested rows or columns out of range.")

    # Task 24: Combined .iloc select range rows 2->6 (iloc[1:6]) and columns 2->4 (iloc[:,1:4])
    show_title("Task 24: Combined .iloc rows iloc[1:6], cols iloc[:,1:4]")
    try:
        print(df.iloc[1:6, 1:4].to_string())
    except Exception as e:
        print("Error:", e)

    # Task 25: Add a new row
    show_title("Task 25: Add a new row (appending at the end)")
    # Make a new row dict: fill known columns, others NaN
    new_index = df.index.max() + 1 if len(df)>0 else 0
    new_row = {c: np.nan for c in df.columns}
    # try to set some sensible example values if columns exist
    if cols_req.get("city"): new_row[cols_req["city"]] = "NEWCITY"
    if cols_req.get("price"): new_row[cols_req["price"]] = 123456
    if cols_req.get("street"): new_row[cols_req["street"]] = "Example Street 1"
    df = pd.concat([df, pd.DataFrame([new_row], index=[new_index])])
    print("Added new row at index", new_index)
    print(df.loc[[new_index]].to_string())

    # Task 26: delete row with index 2
    show_title("Task 26: Delete row with index label 2")
    if 2 in df.index:
        df = df.drop(index=2)
        print("Deleted index 2. Current index list (first 10):", list(df.index)[:10])
    else:
        print("Index 2 not present, nothing deleted.")

    # Task 27: delete row with index from 4 to 7th row (labels 4,5,6,7)
    show_title("Task 27: Delete rows with labels 4 to 7 (if present)")
    labels_to_drop = [lbl for lbl in range(4,8) if lbl in df.index]
    if labels_to_drop:
        df = df.drop(index=labels_to_drop)
        print("Dropped labels:", labels_to_drop)
    else:
        print("None of labels 4..7 present, no rows dropped.")

    # Task 28: Delete 'house_size' column
    show_title("Task 28: Delete column 'house_size'")
    hs_col = cols_req.get("house_size")
    if hs_col and hs_col in df.columns:
        df = df.drop(columns=[hs_col])
        print(f"Column '{hs_col}' dropped.")
    else:
        print("house_size column not found or already dropped.")

    # Task 29: Delete 'house_size' and 'state'
    show_title("Task 29: Delete house_size and state columns (if exist)")
    cols_to_drop = []
    if cols_req.get("house_size") and cols_req.get("house_size") in df.columns:
        cols_to_drop.append(cols_req.get("house_size"))
    if cols_req.get("state") and cols_req.get("state") in df.columns:
        cols_to_drop.append(cols_req.get("state"))
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print("Dropped columns:", cols_to_drop)
    else:
        print("None of specified columns found to drop.")

    # Task 30: Rename column 'state' to 'state_Changed'
    show_title("Task 30: Rename column 'state' -> 'state_Changed'")
    if cols_req.get("state") and cols_req.get("state") in df.columns:
        df = df.rename(columns={cols_req.get("state"): "state_Changed"})
        print("Renamed column. Columns now:", list(df.columns))
    else:
        print("state column not found, cannot rename.")

    # Task 31: Rename index label from 3 to 5
    show_title("Task 31: Rename index label 3 -> 5")
    if 3 in df.index:
        if 5 in df.index:
            print("Warning: label 5 already exists; renaming may create duplicate indices. Proceeding anyway.")
        df = df.rename(index={3:5})
        print("Index labels sample (first 20):", list(df.index)[:20])
    else:
        print("Index label 3 not present; cannot rename.")

    # Task 32: query() to select price < 127400 and city != 'Adjuntas'
    show_title("Task 32: Query price < 127400 and city != 'Adjuntas'")
    if price_col and city_col:
        res = df.query(f"`{price_col}` < 127400 and `{city_col}` != 'Adjuntas'")
        print(res.to_string() if not res.empty else "No rows match query.")
    else:
        print("Price or City column missing for query.")

    # Task 33: sort DataFrame by price ascending
    show_title("Task 33: Sort by price ascending")
    if price_col:
        sorted_df = df.sort_values(by=price_col, ascending=True)
        print(sorted_df[[price_col]].head(10).to_string())
    else:
        print("Price column missing.")

    # Task 34: group by city and sum price
    show_title("Task 34: Group by city and sum of price")
    if city_col and price_col:
        grp = df.groupby(city_col)[price_col].sum().reset_index().sort_values(by=price_col, ascending=False)
        print(grp.to_string(index=False))
    else:
        print("City or Price missing; cannot group.")

    # Task 35: dropna() remove rows with any missing values
    show_title("Task 35: dropna() - remove rows with any NaN")
    df_dropna = df.dropna(how='any').reset_index(drop=True)
    print("After dropna, rows:", len(df_dropna))
    print(df_dropna.head(5).to_string())

    # Task 36: fill NaN values with 0
    show_title("Task 36: fillna(0) - fill NaN with 0")
    df_filled = df.fillna(0)
    print("After fillna(0), sample (first 5 rows):")
    print(df_filled.head(5).to_string())

    show_title("All tasks completed (script finished).")
    print("Notes:")
    print("- If any column names were not found (city/street/zip_code), open the CSV and note exact header text.")
    print("- If you want me to run this and send the actual outputs, upload the CSV here or give the raw GitHub URL.")

if __name__ == "__main__":
    main()
