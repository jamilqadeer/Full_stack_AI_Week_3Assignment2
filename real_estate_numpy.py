# real_estate_numpy_full.py
# Put this file in the same folder as RealEstate-USA.csv
# OR set CSV to a raw GitHub URL (uncomment and replace).

import numpy as np
import pandas as pd
from scipy import stats

# ---------- CONFIG ----------
CSV = "RealEstate-USA.csv"
# If you prefer to read directly from GitHub raw URL, replace above with:
CSV = "https://raw.githubusercontent.com/ShahzadSarwar10/Fullstack-WITH-AI-B-3-SAT-SUN-6Months-Explorer/refs/heads/main/DataSetForPractice/RealEstate-USA.csv"
# ----------------------------

def safe_load(csv_path):
    df = pd.read_csv(csv_path)
    print("Columns in file:", list(df.columns))
    # Normalize and coerce numeric columns
    for col in ["price", "house_size", "acre_lot"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def describe_numeric(arr, name):
    s = pd.Series(arr).dropna()
    print(f"\n--- {name} (n={len(s)}) ---")
    if len(s)==0:
        print("No numeric data.")
        return
    print("Mean:", s.mean())
    print("Median:", s.median())
    m = s.mode()
    print("Mode(s):", m.tolist())
    print("Std (population ddof=0):", s.std(ddof=0))
    if len(s)>1:
        print("Std (sample ddof=1):", s.std(ddof=1))
    print("Variance:", s.var(ddof=0))
    print("Min:", s.min(), "Max:", s.max(), "Range:", s.max()-s.min())
    print("25%,50%,75%:", s.quantile([0.25,0.5,0.75]).to_dict())

def main():
    df = safe_load(CSV)

    # Check required columns
    required = ["brokered_by","price","acre_lot","city","house_size"]
    for r in required:
        if r not in df.columns:
            print(f"Warning: column '{r}' not found in CSV. Please check header names.")
    # Drop rows where essential numeric columns missing (recommended)
    df_clean = df.dropna(subset=["price","house_size"]).reset_index(drop=True)

    # 1. Create numpy arrays
    brokered_by = df_clean["brokered_by"].to_numpy() if "brokered_by" in df_clean.columns else None
    price = df_clean["price"].to_numpy() if "price" in df_clean.columns else None
    acre_lot = df_clean["acre_lot"].to_numpy() if "acre_lot" in df_clean.columns else None
    city = df_clean["city"].to_numpy() if "city" in df_clean.columns else None
    house_size = df_clean["house_size"].to_numpy() if "house_size" in df_clean.columns else None

    print("\nSample (first 5 rows):")
    if brokered_by is not None: print("brokered_by:", brokered_by[:5])
    if price is not None: print("price:", price[:5])
    if acre_lot is not None: print("acre_lot:", acre_lot[:5])
    if city is not None: print("city:", city[:5])
    if house_size is not None: print("house_size:", house_size[:5])

    # 2. Stats on price (mode, median, sd, variance, etc.)
    if price is not None:
        describe_numeric(price, "PRICE")

    # 3. Stats on house_size
    if house_size is not None:
        describe_numeric(house_size, "HOUSE_SIZE")

    # 4. Arithmetic operations (element-wise) - show first 10 results
    if price is not None and house_size is not None:
        print("\n=== Arithmetic (first 10 items) ===")
        print("Add (+) operator:", (price + house_size)[:10])
        print("Add np.add:", np.add(price, house_size)[:10])
        print("Sub (-):", (price - house_size)[:10])
        print("Subtract np.subtract:", np.subtract(price, house_size)[:10])
        print("Mul (*):", (price * house_size)[:10])
        print("Multiply np.multiply:", np.multiply(price, house_size)[:10])

    # 5. Create 2D array (observations x features) => shape (N, 2)
    if price is not None and house_size is not None:
        arr2d = np.column_stack((price, house_size))  # rows = observations, cols = [price, house_size]
        print("\n2D array shape (N,2):", arr2d.shape)
        print("2D first 6 rows:\n", arr2d[:6])

    # 6. Create 3D array using house_size, price, acre_lot
    if house_size is not None and price is not None and acre_lot is not None:
        base = np.column_stack((house_size, price, acre_lot))  # (N,3)
        arr3d = base.reshape(base.shape[0], base.shape[1], 1)   # (N,3,1)
        print("\n3D array shape (N,3,1):", arr3d.shape)
        print("3D sample (first 3):\n", arr3d[:3])
    else:
        print("\n3D array skipped (missing one of house_size, price, acre_lot).")

    # 7. Iterate price with np.nditer
    if price is not None:
        print("\nIterate price with np.nditer (first 20):")
        cnt=0
        for x in np.nditer(price):
            print(x, end=" ")
            cnt+=1
            if cnt>=20: break
        print()

    # 8. Iterate price with np.ndenumerate
    if price is not None:
        print("\nIterate price with np.ndenumerate (first 20):")
        cnt=0
        for idx,val in np.ndenumerate(price):
            print(f"{idx}: {val}")
            cnt+=1
            if cnt>=20: break

    # 9. Seven properties of price array
    if price is not None:
        print("\n=== 7 properties of price array ===")
        print("ndim:", price.ndim)
        print("shape:", price.shape)
        print("size:", price.size)
        print("dtype:", price.dtype)
        print("itemsize:", price.itemsize)
        print("nbytes:", price.nbytes)
        print("strides:", price.strides)

    # 10. Slice arr2d: rows 1st to 3rd, cols 2nd to 4th  (1-based -> 0-based: rows 0:3, cols 1:4)
    if 'arr2d' in locals():
        r0,r1 = 0,3
        c0,c1 = 1,4
        # safe bounds
        r1_safe = min(r1, arr2d.shape[0])
        c1_safe = min(c1, arr2d.shape[1])
        print(f"\nQ10 slice rows[{r0}:{r1}] cols[{c0}:{c1}] -> used rows[{r0}:{r1_safe}] cols[{c0}:{c1_safe}]")
        print(arr2d[r0:r1_safe, c0:c1_safe])

    # 11. Slice arr2d: rows 2nd to 8th, cols 3rd to 5th (1-based -> 0-based: rows 1:8, cols 2:5)
    if 'arr2d' in locals():
        r0,r1 = 1,8
        c0,c1 = 2,5
        r1_safe = min(r1, arr2d.shape[0])
        c1_safe = min(c1, arr2d.shape[1])
        print(f"\nQ11 slice rows[{r0}:{r1}] cols[{c0}:{c1}] -> used rows[{r0}:{r1_safe}] cols[{c0}:{c1_safe}]")
        # if c0 >= arr2d.shape[1] -> empty
        if c0 >= arr2d.shape[1]:
            print("Requested columns start beyond available columns. Result is empty.")
        else:
            print(arr2d[r0:r1_safe, c0:c1_safe])

    # 12. Geometric operations on arr2d: sin, cos, tan, exp, log, sqrt (first 6 rows)
    if 'arr2d' in locals():
        arrf = arr2d.astype(float)
        print("\nGeometric operations (first 6 rows):")
        print("sin:\n", np.sin(arrf)[:6])
        print("cos:\n", np.cos(arrf)[:6])
        print("tan:\n", np.tan(arrf)[:6])
        # safe exp (clip large inputs)
        with np.errstate(over='ignore', invalid='ignore'):
            safe_exp = np.exp(np.clip(arrf, a_min=None, a_max=700))
            print("exp (clipped):\n", safe_exp[:6])
        safe_log = np.where(arrf>0, np.log(arrf), np.nan)
        print("log (NaN for non-positive):\n", safe_log[:6])
        safe_sqrt = np.where(arrf>=0, np.sqrt(arrf), np.nan)
        print("sqrt (NaN for negative):\n", safe_sqrt[:6])

    print("\n--- Script finished ---")

if __name__ == "__main__":
    main()
