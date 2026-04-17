import sys
import os
import random
import time
import zipfile
from datetime import datetime
import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

# 将项目根目录和业务子目录加入路径，确保生产代码及其子包可被导入
project_root = "/Users/xiz/Quant_trade/quant_data"
collector_dir = os.path.join(project_root, "news_collectors/gdelt")

for path in [project_root, collector_dir]:
    if path not in sys.path:
        sys.path.append(path)

# 导入生产环境的逻辑
try:
    from historical_collector import (
        read_gkg_tsv, 
        process_file_task, 
        RuleManager,
        FILES_DIR
    )
except ImportError as e:
    # 尝试备用导入路径
    try:
        from news_collectors.gdelt.historical_collector import (
            read_gkg_tsv, 
            process_file_task, 
            RuleManager
        )
    except:
        print(f"❌ Failed to import production logic even with path adjustments: {e}")
        sys.exit(1)

# ============================
# 配置与常量
# ============================
BASE_DIR = "/Volumes/data24T/docker-volumes/gdelt_cache/files"
SYMBOL_RULES_DIR = os.path.join(project_root, "news_collectors/gdelt/company_rules")

TECH_26 = [
    "TSM", "ASML", "CRM", "PLTR", "NOW", "ADBE", "NFLX", "UBER", 
    "SNOW", "MDB", "PANW", "CRWD", "SMCI", "AMAT", "LRCX", "KLAC", 
    "TXN", "ADI", "MCHP", "ORCL", "FTNT", "ABNB", "CSCO", "IBM", 
    "DELL", "INTU"
]

class RuleValidatorAgent:
    def __init__(self, symbols=None):
        self.symbols = symbols or TECH_26
        self.rule_manager = RuleManager(SYMBOL_RULES_DIR)
        self.sample_files = self.get_random_samples(10)
        
    def get_random_samples(self, count):
        all_zips = glob(os.path.join(BASE_DIR, "*.gkg.csv.zip"))
        if not all_zips: return []
        # 选取几个不同年份的样本
        return random.sample(all_zips, min(count, len(all_zips)))

    def analyze_with_slm(self, symbol, snippet):
        url = "http://localhost:1234/v1/chat/completions"
        prompt = f"Is the news relevant to technologies of '{symbol}'? Answer YES or NO only.\nSnippet: {snippet[:300]}"
        
        try:
            resp = requests.post(url, json={
                "model": "qwen3.5-4b",
                "messages": [
                    {"role": "system", "content": "Answer ONLY 'YES' or 'NO'."},
                    {"role": "user", "content": f"/no_think\n{prompt}"}
                ],
                "temperature": 0,
                "max_tokens": 5, 
                "include_thoughts": False
            }, timeout=10)
            res_text = resp.json()['choices'][0]['message']['content'].strip().upper()
            
            is_relevant = "YES" in res_text
            return {
                "relevant": is_relevant,
                "suggestions": [], # 极速模式下不再请求建议
                "reason": "Binary Audit"
            }
        except:
            return {"relevant": False, "reason": "Audit Failed"}

    def auto_update_json(self, symbol, analysis):
        """双向调节：既能扩充（发现新词），也能缩减（添加负向过滤）"""
        rule_path = os.path.join(SYMBOL_RULES_DIR, f"{symbol}.json")
        if not analysis or not os.path.exists(rule_path): return None
        
        try:
            with open(rule_path, 'r') as f: data = json.load(f)
            updated = False
            
            # 1. 扩充逻辑 (If Relevant)
            if analysis.get("relevant"):
                existing = set([k.lower() for k in data.get("expansion_keywords", []) + data.get("primary_keywords", [])])
                suggestions = analysis.get("suggestions", [])
                added_kws = [k for k in suggestions if k.lower() not in existing and len(k)>3]
                if added_kws:
                    data.setdefault("expansion_keywords", []).extend(added_kws)
                    updated = True
            
            # 2. 缩减/排除逻辑 (If NOISE)
            else:
                neg_suggestions = analysis.get("negative_suggestions", [])
                if neg_suggestions:
                    existing_neg = set([k.lower() for k in data.get("negative_keywords", [])])
                    new_negs = [k for k in neg_suggestions if k.lower() not in existing_neg]
                    if new_negs:
                        data.setdefault("negative_keywords", []).extend(new_negs)
                        updated = True

            if updated:
                with open(rule_path, 'w') as f: json.dump(data, f, indent=4)
                return added_kws if 'added_kws' in locals() else True
        except: pass
        return False

    def run_validation_cycle(self, max_rounds=2):
        print(f"🚀 [Agent] Multi-Round Validation with Bidirectional Tuning")
        
        # 准备公司列表（符合生产环境格式）
        companies = [{"symbol": s} for s in self.symbols]
        rule_manager = self.rule_manager
        
        for r_idx in range(1, max_rounds + 1):
            print(f"\n🔔 --- ROUND {r_idx} ---")
            
            # --- 模拟生产环境的关键词初始化逻辑 ---
            all_keywords_map = {}
            case_insensitive_keywords = []
            for company in companies:
                symbol = company['symbol']
                keywords = rule_manager.get_keywords(symbol, datetime.now())
                if not keywords: keywords = [symbol]
                for k in keywords:
                    if not isinstance(k, str) or len(k) <= 1: continue
                    k_lower = k.lower()
                    if k_lower not in all_keywords_map: all_keywords_map[k_lower] = []
                    all_keywords_map[k_lower].append(symbol)
                    case_insensitive_keywords.append(re.escape(k_lower))
            
            global_pattern = "|".join(case_insensitive_keywords)
            if not global_pattern:
                print("⚠️ No keywords available for current rules.")
                continue

            all_reports = []
            zero_hit_symbols = set(self.symbols)
            symbol_to_company = {s: {"symbol": s} for s in self.symbols}
            
            # 调用生产环境的扫描逻辑
            from news_collectors.gdelt.historical_collector import process_file_task as prod_scanner
            for i, zip_path in enumerate(self.sample_files):
                fname = os.path.basename(zip_path)
                print(f"   📂 [{i+1}/{len(self.sample_files)}] Scanning {fname}...")
                
                res = prod_scanner(
                    zip_path, 
                    rule_manager, 
                    all_keywords_map, 
                    global_pattern, 
                    symbol_to_company, 
                    None, # progress_lock
                    {}
                ) 
                
                if res and res.get("matches"):
                    print(f"      ✅ Found {len(res['matches'])} production matches.")
                    for m in res["matches"]:
                        sym = m["company"]["symbol"]
                        if sym in zero_hit_symbols: zero_hit_symbols.remove(sym)
                        
                        # 🔍 审计：判断这条新闻是靠什么抓到的
                        title = str(m["row"].get("Title", "")).lower()
                        source_tag = "Entity_Col"
                        url_val = str(m["row"].get("URL", "")).lower()
                        if any(k.lower() in url_val for k in all_keywords_map):
                            source_tag = "URL_Slug"
                        
                        all_reports.append({
                            "symbol": sym,
                            "snippet": title + " " + str(m.get("content", "")),
                            "url": m["row"].get("URL"),
                            "source_tag": source_tag
                        })
                else:
                    print(f"      ⚪ No matches found in this file.")
            
            # 💡 增加：对 0 HITS 的标的进行“模糊探测”
            if zero_hit_symbols:
                print(f"🔎 [Agent] No hits for {len(zero_hit_symbols)} symbols. Attempting Fuzzy Probe for missing news...")
                for z_sym in list(zero_hit_symbols):
                    rule = self.rule_manager.rules.get(z_sym)
                    if not rule: continue
                    full_name = getattr(rule, "company_name", z_sym)
                    
                    found_any = False
                    for zip_path in self.sample_files[:5]:
                        try:
                            with zipfile.ZipFile(zip_path) as z:
                                with z.open(z.namelist()[0]) as f:
                                    import io, csv
                                    reader = csv.DictReader(io.TextIOWrapper(f, encoding="ISO-8859-1"), delimiter='\t')
                                    for row in reader:
                                        orgs = row.get("V2Organizations", "") or row.get("Organizations", "")
                                        if full_name.lower() in orgs.lower():
                                            url = row.get("DocumentIdentifier") or row.get("URL")
                                            all_reports.append({
                                                "symbol": z_sym,
                                                "snippet": f"Fuzzy Match: {full_name} | Orgs: {orgs[:150]}",
                                                "url": url,
                                                "source_tag": "Fuzzy_Probe"
                                            })
                                            found_any = True
                                            break
                        except: pass
                        if found_any: break
                    if not found_any:
                        print(f"      ⚪ {z_sym} still quiet even with full name search.")

            # 🧠 全量并行审计阶段
            audit_log = []
            updates_log = {}
            if all_reports:
                print(f"🕵️ [Agent] Starting Concurrent Full Audit on all {len(all_reports)} captures...")
                
                def audit_task(sample):
                    res = self.analyze_with_slm(sample["symbol"], sample["snippet"])
                    if res:
                        res["source_tag"] = sample["source_tag"]
                        res["symbol"] = sample["symbol"]
                        res["snippet"] = sample["snippet"]
                    return res, sample

                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(audit_task, s) for s in all_reports]
                    for i, future in enumerate(futures):
                        res, sample = future.result()
                        if not res: continue
                        audit_log.append(res)
                        
                        # 进度汇报
                        if (i + 1) % 10 == 0:
                            print(f"      🗂   Progress: [{i+1}/{len(all_reports)}] audited...")
                            
                        if res.get("relevant"):
                            added_kws = self.auto_update_json(sample["symbol"], res)
                            if added_kws:
                                updates_log[sample["symbol"]] = updates_log.get(sample['symbol'], []) + added_kws

            self.print_audit_summary(audit_log)
            self.print_summary(all_reports, updates_log, audit_log)

            # 💾 持久化“真家伙”供用户后续翻看
            import json
            save_path = "/Users/xiz/Quant_trade/quant_data/audit_samples_latest.json"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "total_captures": len(all_reports),
                    "articles": all_reports
                }, f, indent=4, ensure_ascii=False)
            print(f"\n💾 [Persistence] All captured articles saved to: {save_path}")

    def print_audit_summary(self, audit_log):
        # 统计不同来源的“准确率”
        stats = {} # {tag: {"relevant": 0, "total": 0}}
        for entry in audit_log:
            tag = entry.get("source_tag", "Unknown")
            if tag not in stats: stats[tag] = {"relevant": 0, "total": 0}
            stats[tag]["total"] += 1
            # 兼容不同格式的 relevant 判定
            rel = entry.get("relevant")
            if isinstance(rel, str): rel = rel.lower() == "true"
            if rel: stats[tag]["relevant"] += 1
        
        print("\n" + "🔍" * 5 + " MATCH QUALITY AUDIT REPORT " + "🔍" * 5)
        print(f"{'Source Tag':<15} | {'Precision':<10} | {'Count'}")
        print("-" * 45)
        for tag, data in sorted(stats.items()):
            precision = (data["relevant"] / data["total"] * 100) if data["total"] > 0 else 0
            print(f"{tag:<15} | {precision:>8.1f}% | {data['total']}")
        print("=" * 50)

    def print_summary(self, reports, updates_log, audit_log=[]):
        counts = {}
        samples = {}
        for r in reports:
            s = r["symbol"]
            counts[s] = counts.get(s, 0) + 1
            if s not in samples: samples[s] = []
            if len(samples[s]) < 2: samples[s].append(r)
            
        print("\n" + "="*60)
        print(f"{'SYM':<8} | {'HITS':<6} | {'SAMPLES (Correctness Verification)'}")
        print("-" * 60)
        processed = 0
        for s in sorted(counts.keys(), key=lambda x: counts[x], reverse=True):
            status = "✨ UPDATED" if s in updates_log else "OK"
            print(f"{s:<8} | {counts[s]:<6} | {status}")
            for smp in samples[s]:
                # 🔍 从审计日志中寻找对应的 AI 判断
                judgment = "N/A"
                reason = "Not Audited"
                for entry in audit_log:
                    if entry.get("symbol") == s and entry.get("snippet") == smp["snippet"]:
                        judgment = "✅ RELEVANT" if entry.get("relevant") else "❌ NOISE"
                        reason = entry.get("reason", "No reason")
                        break
                
                snippet = smp['snippet'].replace('\n', ' ')[:80]
                url = smp['url'][:60]
                print(f"         └─ [{judgment}] {reason[:60]}...")
                print(f"         └─ Context: {snippet}...")
                print(f"         └─ URL: {url}")
            processed += 1
            if processed > 15: break
            
        if updates_log:
            print("\n💡 [Auto-Evolution Summary]:")
            for s, ks in list(updates_log.items())[:5]:
                print(f"   - {s}: Added {len(ks)} new keywords: {', '.join(ks[:3])}...")
        print("="*60)

if __name__ == "__main__":
    agent = RuleValidatorAgent()
    agent.run_validation_cycle(max_rounds=2)
