import zipfile
import pandas as pd
import os
import glob

def view_captured_articles():
    base_dir = "/Volumes/data24T/docker-volumes/gdelt_cache/files"
    all_zips = sorted(glob.glob(os.path.join(base_dir, "*.gkg.csv.zip")))
    sample_files = all_zips[:10]  # 取前10个作为展示
    
    symbols = ["IBM", "ABNB", "TXN", "AAPL", "CRM", "TSM", "NVDA", "SMCI"]
    
    print(f"\n{'SYMBOL':<8} | {'URL SOURCE'}")
    print("-" * 60)
    
    found_count = 0
    for p in sample_files:
        try:
            with zipfile.ZipFile(p) as z:
                with z.open(z.namelist()[0]) as f:
                    # GDELT GKG 数据通常没有 Header
                    df = pd.read_csv(f, sep='\t', header=None, engine='c', low_memory=False, quoting=3)
                    
                    # 我们重点搜 4 (URL), 7 (Persons), 9 (Organizations)
                    for sym in symbols:
                        # 简单的模糊匹配用于展示
                        mask = (df[4].astype(str).str.contains(sym, case=False, na=False) | 
                                df[9].astype(str).str.contains(sym, case=False, na=False))
                        
                        matches = df[mask]
                        for url in matches[4].head(3):
                            print(f"{sym:<8} | {url}")
                            found_count += 1
        except Exception as e:
            continue
            
    print("-" * 60)
    print(f"Total samples found in 10 files: {found_count}")

if __name__ == "__main__":
    view_captured_articles()
