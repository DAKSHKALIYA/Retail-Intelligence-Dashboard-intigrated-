"""Run this first if something does not work:   python check_setup.py"""
import importlib
import os
import sys

print(f"Python version : {sys.version.split()[0]}")
ok = True

f = "Online Retail.xlsx"
if not os.path.exists(f):
    print(f"[X] '{f}' not found in this folder. Put it next to streamlit_app.py.")
    ok = False
else:
    size = os.path.getsize(f)
    head = open(f, "rb").read(4)
    if size < 1_000_000 or head != b"PK\x03\x04":
        print(f"[X] '{f}' is broken (size={size} bytes). Copy the original file again (about 23 MB).")
        ok = False
    else:
        print(f"[OK] {f} ({size/1e6:.1f} MB)")

for mod in ["pandas", "numpy", "openpyxl", "plotly", "streamlit", "sklearn", "joblib", "requests"]:
    try:
        importlib.import_module(mod)
        print(f"[OK] {mod}")
    except ImportError:
        print(f"[X] {mod} missing -> activate venv and run: pip install -r requirements.txt")
        ok = False

for p in ["ml_output/customer_scores.csv", "rfm_customer_table.csv"]:
    print(("[OK] " if os.path.exists(p) else "[X] ") + p)
    ok = ok and os.path.exists(p)

print("\nAll good. Run: streamlit run streamlit_app.py" if ok else "\nFix the [X] items above, then run this again.")
