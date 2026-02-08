#!/usr/bin/env python3
"""
Ambiguous Names Rule Handler
处理歧义名称的特殊规则
"""

import re
import os
from datetime import datetime
from .base_rule import BaseRule

# 延迟导入 SLM Filter（避免循环依赖）
_slm_filter = None

def get_slm_filter():
    """获取全局 SLM 过滤器实例"""
    global _slm_filter
    if _slm_filter is None:
        try:
            from .slm_filter import SLMFilter
            # 检查是否启用 SLM（通过环境变量控制）
            use_slm = os.getenv("USE_SLM_FILTER", "true").lower() == "true"
            _slm_filter = SLMFilter(enabled=use_slm)
        except Exception as e:
            print(f"⚠️ SLM Filter 初始化失败: {e}，将使用纯规则过滤")
            _slm_filter = None
    return _slm_filter


class AmbiguousNameRule(BaseRule):
    """歧义名称规则"""
    
    # 定义静态排除逻辑：秒杀已知的噪音 (Fast-Kill)
    STATIC_KILL_PATTERNS = {
        "MSFT": [
            r"box\s+office", r"post\s+office", r"sheriff\'?s\s+office", 
            r"governor\'?s\s+office", r"mayor\'?s\s+office", r"police\s+office",
            r"dental\s+office", r"doctor\'?s\s+office", r"office\s+building",
            r"box-office", r"\boffice\s+space\b", r"coroner\'?s\s+office",
            r"clerk\'?s\s+office", r"prosecutor\'?s\s+office"
        ],
        "ARM": [
            r"armed\s+(man|robbery|suspect|person|forces|conflict|gang|group|citizen)", 
            r"arms\s+(race|deal|control|sales|shipment|trafficking)", r"heavy\s+arms",
            r"firearms", r"took\s+up\s+arms", r"under\s+arms", r"\barm\s+in\s+arm\b",
            r"broken\s+arm", r"arm\s+injury"
        ],
        "META": [
            r"meta-analysis", r"meta-search", r"meta-data", r"meta-theory",
            r"metastatic", r"metabolism", r"meta-tags",
            r"armed", r"crime", r"arrest", r"police", r"murder", r"robbery"
        ],
        "AAPL": [
            r"big\s+mac", r"mcdonald\'?s", r"happy\s+meal", r"burger", r"mcnuggets",
            r"apple\s+pie", r"apple\s+juice", r"apple\s+orchard", r"cooking", r"recipes",
            r"fruit", r"vegetables", r"grocery", r"supermarket",
            r"apple\s+design", r"recipe\s+for"
        ],
        "AMD": [
            r"box\s+office", r"post\s+office", r"sheriff\'?s", r"cinema", r"movie", r"film",
            r"amd\s+intel" 
        ],
        "META": [
            r"arrest", r"police", r"murder", r"crime", r"armed", r"suspect",
            r"shooting", r"killed", r"victim", r"jail", r"prison",
            r"share\b.*\bfacebook", r"follow\b.*\bfacebook", r"log\b.*\bfacebook",
            r"facebook\.com/(login|register|pages|groups)", 
            r"/sharer/sharer\.php", r"\?u=.*facebook\.com",
            r"bikini", r"celeb", r"gossip", r"boyfriend", r"girlfriend", r"dating",
            r"red\s+carpet", r"vacation", r"family\s+getaway", r"top\s+nine",
            r"showbiz", r"paparazzi", r"engagement", r"wedding", r"pregnan", r"baby\s+bump"
        ],
        "AMZN": [
            r"deal\s+of\s+the\s+day", r"best\s+seller", r"kindle\s+edition",
            r"paperback", r"hardcover", r"gift\s+card", r"sign\s+up\s+for\s+prime",
            r"add\s+to\s+cart", r"customer\s+review", r"in\s+stock"
        ],
        "GOOGL": [
            r"google\s+play", r"google\s+maps", r"google\s+images", 
            r"sign\s+in\s+with\s+google", r"google\s+account",
            r"google\s+search", r"googled\b.*\bdiets",
            r"google\.com/(maps|search|amp)", r"accounts\.google\.com"
        ],
        "MU": [
            r"murder", r"muslim", r"municipality", 
            r"manchester\s+united", r"make\s+up", r"makeup", r"mukesh"
        ],
        "DDOG": [
            r"drug", r"police", r"crime", r"arrest", r"sheriff", r"suspect",
            r"dog\s+bite", r"stray\s+dog", r"pet\s+food"
        ]
    }

    def __init__(self, symbol, config, company_name=None, full_config=None):
        super().__init__(symbol, config)
        self.company_name = company_name or symbol
        self.problem = config.get('problem', '')
        self.required_keywords = config.get('required_keywords', [])
        
        # 🚀 从全局配置中获取扩展词，用于强匹配
        full_config = full_config or {}
        self.primary_keywords = full_config.get('primary_keywords', [])
        self.expansion_keywords = full_config.get('expansion_keywords', [])
        self.strict_validation_keywords = full_config.get('strict_validation_keywords', []) # 🆕 从JSON加载强校验词
        
        self.min_matches = config.get('min_matches', 1)
        self.exclude_patterns = config.get('exclude_patterns', [])
        self.case_sensitive = config.get('case_sensitive', False)
        self.use_slm = config.get('use_slm', True)
    
    def get_keywords(self, article_date):
        """返回必需的关键词"""
        return self.required_keywords
    
    def should_include(self, article):
        title = article.get('title', '') or ''
        content = article.get('content', '') or ''
        # 回归稳健：不再将 URL 混入校验，防止 mu 在 /music/ 这种路径中误命中
        full_text = title + ' ' + content
        
        # 0. 防止空内容
        if not full_text.strip():
            return False

        # 2. 静态黑名单 (Static Kill)
        if self.symbol in self.STATIC_KILL_PATTERNS:
            for pattern in self.STATIC_KILL_PATTERNS[self.symbol]:
                if re.search(pattern, full_text, re.IGNORECASE):
                    # 记录静默拦截
                    print(f"🚫 Static Kill: {self.symbol} - Pattern: {pattern}")
                    return False

        # 3. 关键词匹配计数 (必需关键词)
        matches = []
        for keyword in self.required_keywords:
            # 对于短 Ticker (<=4)，必须严格词边界；对于长词可稍微放宽
            if len(keyword) <= 4:
                found = re.search(r'\b' + re.escape(keyword) + r'\b', full_text, re.IGNORECASE)
            else:
                found = True if keyword.lower() in full_text.lower() else False
            
            if found:
                matches.append(keyword)
        
        if len(matches) < self.min_matches:
            # print(f"❌ Keywords Match FAIL: {self.symbol} - Found {len(matches)}/{self.min_matches} required keywords. Title: {title[:60]}")
            return False

        # 4. 特殊短Ticker强校验 (Mandatory Identity Check)
        if self.strict_validation_keywords:
            has_identity = False
            full_text_lower = full_text.lower()
            matched_identity = []
            for identity in self.strict_validation_keywords:
                if identity.lower() in full_text_lower:
                    has_identity = True
                    matched_identity.append(identity)
                    # break # 不跳出，为了打印所有命中的词
            
            if not has_identity:
                print(f"🛡️ Strict Identity BLOCK: {self.symbol} - Missing mandatory context. Title: {title[:60]}")
                return False
            else:
                print(f"✅ Strict Identity PASS: {self.symbol} - Matched: {matched_identity} - Title: {title[:60]}")

        # 5. 强匹配逻辑：如果命中极其特殊的产品词，直接放行 Bypassing SLM
        # 🆕 改进：从 strong_keywords 中排除过于大众的词，防止其绕过 SLM 检查
        too_common_to_bypass = ["Instagram", "Facebook", "WhatsApp", "YouTube", "Android", "Chrome", "Google", "Amazon", "Apple", "Microsoft"]
        strong_keywords = [
            k for k in self.expansion_keywords 
            if len(k) > 3 
            and k.lower() not in [pk.lower() for pk in self.primary_keywords]
            and k not in too_common_to_bypass
        ]
        if strong_keywords:
            # 优先匹配长词
            strong_keywords.sort(key=len, reverse=True)
            strong_pattern = r'\b(' + '|'.join(re.escape(k) for k in strong_keywords) + r')\b'
            if re.search(strong_pattern, full_text, re.IGNORECASE):
                return True

        # 6. SLM 智能判断
        # 修改：即使没有正文，也可以用SLM检查标题
        has_title = len(title.strip()) > 10
        has_content = len(content) > 30 and not content.startswith("http")
        
        # 只要有标题或有内容，就可以使用SLM
        if self.use_slm and (has_title or has_content):
            slm = get_slm_filter()
            if slm:
                # 调用 SLM (注意：不带 reason 返回)
                is_relevant = slm.is_relevant(self.symbol, self.company_name, title, content)
                
                # 记录拦截日志
                log_dir = "/home/xiz/logs/history_collector"
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, "slm_interceptions.log")
                
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        status = "PASSED" if is_relevant else "INTERCEPTED"
                        log_entry = (
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"Result: {status}\n"
                            f"Symbol: {self.symbol}\n"
                            f"Title: {title}\n"
                            f"Content_Length: {len(content)}\n"
                            f"Content: {content[:300]}...\n"
                            f"{'-'*50}\n"
                        )
                        f.write(log_entry)
                        f.flush()
                except Exception as e:
                    print(f"⚠️ Log writing failed: {e}")

                if not is_relevant:
                    print(f"🤖 SLM INTERCEPTED: {self.symbol} - Title: {title[:60]}")
                    return False
                else:
                    print(f"✅ SLM PASS: {self.symbol} - Title: {title[:60]}")

        # 默认放行
        return True
