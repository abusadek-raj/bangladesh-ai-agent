import os
import sqlite3
import pandas as pd
from datasets import load_dataset

os.makedirs("dbs", exist_ok=True)

def prepare_institutions():
    print("Processing Institutions dataset...")
    try:
        ds = load_dataset("Mahadih534/Institutional-Information-of-Bangladesh", split="train")
        df = ds.to_pandas()
    except Exception as e:
        print(f"HuggingFace download failed, checking local CSV: {e}")
        df = pd.read_csv("data/institutions.csv")
    
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    conn = sqlite3.connect("dbs/institutions.db")
    df.to_sql("institutions", conn, if_exists="replace", index=False)
    conn.close()
    print("Created dbs/institutions.db")

def prepare_hospitals():
    print("Processing Hospitals dataset...")
    try:
        ds = load_dataset("Mahadih534/all-bangladeshi-hospitals", split="train")
        df = ds.to_pandas()
    except Exception as e:
        print(f"HuggingFace download failed, checking local CSV: {e}")
        df = pd.read_csv("data/hospitals.csv")
        
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    conn = sqlite3.connect("dbs/hospitals.db")
    df.to_sql("hospitals", conn, if_exists="replace", index=False)
    conn.close()
    print("Created dbs/hospitals.db")

def prepare_restaurants():
    print("Processing Restaurants dataset...")
    try:
        ds = load_dataset("Mahadih534/Bangladeshi-Restaurant-Data", split="train")
        df = ds.to_pandas()
    except Exception as e:
        print(f"HuggingFace download failed, checking local CSV: {e}")
        df = pd.read_csv("data/restaurants.csv")
        
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    conn = sqlite3.connect("dbs/restaurants.db")
    df.to_sql("restaurants", conn, if_exists="replace", index=False)
    conn.close()
    print("Created dbs/restaurants.db")

if __name__ == "__main__":
    prepare_institutions()
    prepare_hospitals()
    prepare_restaurants()
    print("All databases initialized successfully!")