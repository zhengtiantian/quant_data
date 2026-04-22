#!/usr/bin/env python3
"""
Ambiguous Names Rule Handler
处理歧义名称的特殊规则
"""

import re
import os
import threading
from collections import defaultdict
from collections import Counter
from datetime import datetime
from .base_rule import BaseRule

# 延迟导入 SLM Filter（避免循环依赖）
_slm_filter = None
_slm_filter_lock = threading.Lock()
_event_lock = threading.Lock()
_event_counter = defaultdict(int)
_event_type_counter = defaultdict(Counter)
_event_symbol_counter = defaultdict(Counter)

# 每步过滤计数器
_filter_stats_lock = threading.Lock()
_filter_stats = defaultdict(lambda: defaultdict(int))


def track_filter_step(step_name):
    """记录每步过滤的计数（仅累计，不输出）"""
    thread_name = threading.current_thread().name
    owner = thread_name.split("-scan_")[0] if "-scan_" in thread_name else thread_name
    with _filter_stats_lock:
        _filter_stats[owner][step_name] += 1


def print_filter_summary(worker_name=None):
    """输出过滤汇总并重置计数器"""
    owner = worker_name or threading.current_thread().name
    with _filter_stats_lock:
        s = dict(_filter_stats.get(owner, {}))
        _filter_stats[owner] = defaultdict(int)
    if not s:
        return
    entered = s.get("entered", 0)
    if entered == 0:
        return
    after_static = entered - s.get("killed_by_static", 0)
    after_keyword = after_static - s.get("killed_by_keyword", 0)
    after_strict = after_keyword - s.get("killed_by_strict", 0)
    bypassed = s.get("bypassed_by_strong", 0)
    slm_kill = s.get("killed_by_slm", 0)
    slm_pass = s.get("slm_pass", 0)
    default_pass = s.get("default_pass", 0)
    final = bypassed + slm_pass + default_pass
    print(
        f"[{owner}] 📊 Filter summary: {entered} entered "
        f"→ ①static_kill(-{s.get('killed_by_static', 0)})={after_static} "
        f"→ ②keyword(-{s.get('killed_by_keyword', 0)})={after_keyword} "
        f"→ ③strict(-{s.get('killed_by_strict', 0)})={after_strict} "
        f"→ ④strong_bypass={bypassed} "
        f"→ ⑤SLM(pass={slm_pass}, kill={slm_kill}) "
        f"→ ⑥default_pass={default_pass} "
        f"→ ✅ final={final}"
    )


def emit_rule_event(event_type, symbol, detail):
    """规则日志按批聚合输出，避免逐条刷屏。"""
    step = int(os.getenv("RULE_EVENT_EVERY", "5000"))
    thread_name = threading.current_thread().name
    owner = thread_name.split("-scan_")[0] if "-scan_" in thread_name else thread_name
    with _event_lock:
        _event_counter[owner] += 1
        _event_type_counter[owner][event_type] += 1
        _event_symbol_counter[owner][symbol] += 1
        current = _event_counter[owner]
        if current % step != 0:
            return

        top_types = ", ".join(f"{k}:{v}" for k, v in _event_type_counter[owner].most_common(4))
        top_symbols = ", ".join(f"{k}:{v}" for k, v in _event_symbol_counter[owner].most_common(4))
        print(
            f"[{owner}] 🧾 Rule events progress: {current} records processed "
            f"(types: {top_types}; symbols: {top_symbols})"
        )
        print(f"[{owner}] 🧾 Rule events sample: {detail}")

def get_slm_filter():
    """获取全局 SLM 过滤器实例"""
    global _slm_filter
    if _slm_filter is not None:
        return _slm_filter
    with _slm_filter_lock:
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


def _parse_symbol_set(raw_value):
    return {s.strip().upper() for s in raw_value.split(",") if s.strip()}


class AmbiguousNameRule(BaseRule):
    """歧义名称规则"""

    _ABSOLUTE_BYPASS_RAW = {
        "AAPL": [r"tim\s+cook", r"\biphone\b", r"\bipad\b", r"\bmacbook\b", r"\bios\b", r"app\s+store", r"apple\s+vision", r"steve\s+jobs", r"apple\s+watch", r"airpods"],
        "MSFT": [r"satya\s+nadella", r"\bazure\b", r"office\s+365", r"\bmicrosoft\s+365\b", r"\bgame\s+pass\b", r"bill\s+gates", r"surface\s+pro", r"microsoft\s+teams", r"phil\s+spencer"],
        "GOOGL": [r"sundar\s+pichai", r"\byoutube\b", r"\bandroid\b", r"google\s+cloud", r"google\s+ads", r"pixel\s+(phone|watch|tablet)", r"larry\s+page", r"sergey\s+brin", r"waymo"],
        "AMZN": [r"jeff\s+bezos", r"andy\s+jassy", r"amazon\s+web\s+services", r"\baws\b", r"prime\s+video", r"\bkindle\b", r"\balexa\b", r"amazon\s+prime"],
        "META": [r"oculus", r"meta\s+quest", r"facebook\s+ads", r"reality\s+labs", r"threads\s+app"],
        "TSLA": [r"model\s+(s|3|x|y)", r"cybertruck", r"gigafactory", r"tesla\s+autopilot", r"supercharger", r"\btesla\s+energy\b", r"\bmegapack\b", r"\bspacex\b", r"\bstarlink\b", r"\bxai\b"],
        "NVDA": [r"jensen\s+huang", r"geforce", r"rtx\s+\d{4}", r"cuda", r"h100", r"a100", r"nvidia\s+dgx", r"\bwith\s+nvidia\b", r"\bpowered\s+by\s+nvidia\b"],
        "AMD": [r"lisa\s+su", r"ryzen", r"radeon", r"epyc", r"threadripper", r"mi300"],
        "ARM": [r"rene\s+haas", r"arm\s+architecture", r"cortex-", r"neoverse", r"mali\s+gpu"],
        "QCOM": [r"cristiano\s+amon", r"snapdragon", r"\bcdma\b", r"qualcomm\s+hexagon"],
        "AVGO": [r"hock\s+tan", r"\bvmware\b", r"symantec", r"broadcom\s+software"],
        "ORCL": [r"larry\s+ellison", r"safra\s+catz", r"oracle\s+cloud", r"oracle\s+database", r"oracle\s+earnings", r"\bexadata\b", r"\bnetsuite\b"],
        "CSCO": [r"chuck\s+robbins", r"cisco\s+systems", r"\bwebex\b", r"cisco\s+router", r"cisco\s+switch"],
        "IBM": [r"arvind\s+krishna", r"\bred\s+hat\b", r"ibm\s+cloud", r"\bwatson\b", r"ibm\s+mainframe"],
        "DELL": [r"michael\s+dell", r"dell\s+technologies", r"\bpoweredge\b", r"dell\s+emc", r"vmware"],
        "INTU": [r"sasan\s+goodarzi", r"\bturbotax\b", r"\bquickbooks\b", r"\bmailchimp\b", r"intuit\s+platform"],
        "TSM": [r"\btsmc\b", r"cc\s+wei", r"taiwan\s+semiconductor", r"\b3nm\b", r"\b5nm\b", r"\b2nm\b"],
        "MU": [r"sanjay\s+mehrotra", r"\bhbm\b", r"micron\s+technology", r"\bdram\b", r"\bnand\s+flash\b"],
        "AMAT": [r"gary\s+dickerson", r"applied\s+materials", r"semiconductor\s+equipment"],
        "LRCX": [r"tim\s+archer", r"lam\s+research", r"\betch\s+equipment\b"],
        "KLAC": [r"rick\s+wallace", r"\bkla\b", r"kla\s+tencor", r"process\s+control\s+equipment"],
        "MRVL": [r"matt\s+murphy", r"marvell\s+technology", r"marvell\s+semiconductor"],
        "ADI": [r"vincent\s+roche", r"analog\s+devices", r"\blinear\s+technology\b"],
        "NXPI": [r"kurt\s+sievers", r"nxp\s+semiconductors", r"\bnxp\b"],
        "ON": [r"hassane\s+el.khoury", r"\bonsemi\b", r"on\s+semiconductor"],
        "STX": [r"dave\s+mosley", r"seagate\s+technology", r"\bseagate\b"],
        "WDC": [r"david\s+goeckeler", r"western\s+digital", r"\bsandisk\b"],
        "SWKS": [r"liam\s+griffin", r"skyworks\s+solutions", r"skyworks\s+semiconductor"],
        "FTNT": [r"ken\s+xie", r"\bfortigate\b", r"fortinet\s+firewall", r"fortinet\s+security"],
        "ABNB": [r"brian\s+chesky", r"airbnb\s+host", r"airbnb\s+listing", r"airbnb\s+revenue"],
        "INTC": [r"pat\s+gelsinger", r"\bxeon\b", r"\bcore\s+ultra\b", r"intel\s+foundry", r"gaudi\s+accelerator"],
        "MCHP": [r"ganesh\s+moorthy", r"microchip\s+technology", r"\bpic\s+microcontroller\b"],
        "ASML": [r"peter\s+wennink", r"\beuv\b", r"\bduv\b", r"asml\s+scanner", r"extreme\s+ultraviolet"],
    }

    _TOO_COMMON_TO_BYPASS = {"instagram", "facebook", "whatsapp", "youtube", "android", "chrome", "google", "amazon", "apple", "microsoft"}

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

    # 在强直通之前增加“业务主体”过滤，避免平台载体/商品页/外围品牌词直接放行。
    CONTEXTUAL_REJECT_PATTERNS = {
        "AAPL": [
            r"\bac\s+adapter\b", r"\bcharger\b", r"\bcable\b", r"\bsleeve\b", r"\bcase\b",
            r"\bskin\b", r"\bstand\b", r"\bdock\b", r"\bkeyboard\s+cover\b",
            r"\bfor\s+macbook\b.*\b(accessory|adapter|charger|case|sleeve|dock)\b",
            r"\bcompatible\s+with\s+macbook\b",
            r"\brefurbished\b", r"\bbuy\s+now\b", r"\badd\s+to\s+cart\b"
        ],
        "MSFT": [
            r"\brelease\s+.+\s+·\s+.+/.+", r"\bgithub\s+release\b", r"\bappx?\b",
            r"\bdownload\s+for\s+windows\b", r"\bwindows\s+download\b", r"\bpcworld\b",
            r"\bhow\s+to\b.*\bwindows\b", r"\btutorial\b.*\bwindows\b",
            r"\barticles\s+by\b", r"\bfor\s+windows\s+\d+\b",
            r"\bforum\b", r"\bactivation\b", r"\breview\b",
            r"\bnintendo\s+switch\b", r"\bps4\b", r"\bplaystation\b",
            r"\bavailable\s+on\s+xbox\b", r"\bcoming\s+to\s+xbox\b", r"\bon\s+xbox\s+one\b",
            r"\bxbox\s+one\b", r"\bxbox\s+series\s+[xs]\b", r"\bwindows\s+pc\b"
        ],
        "AMZN": [
            r"\badd\s+to\s+cart\b", r"\bbuy\s+now\b", r"\bcurrently\s+unavailable\b",
            r"\bcustomer\s+reviews?\b", r"\blist\s+price\b", r"\bpaperback\b", r"\bhardcover\b",
            r"\bkindle\s+edition\b", r"\bships\s+from\b", r"\bsold\s+by\b",
            r"\bproduct\s+details\b", r"\bin\s+stock\b", r"\bout\s+of\s+stock\b"
        ],
        "GOOGL": [
            r"\bwatch\s+on\s+youtube\b", r"\bsubscribe\b.*\bchannel\b",
            r"\byoutube\s+channel\b", r"\bmusic\s+video\b", r"\bofficial\s+trailer\b",
            r"\breaction\s+video\b", r"\bwalkthrough\b", r"\bgameplay\b",
            r"\bavailable\s+on\s+android\b", r"\bandroid\s+app\b", r"\bapk\b",
            r"\bhow\s+to\b.*\b(android|youtube|chrome)\b", r"\btutorial\b.*\b(android|youtube|chrome)\b"
        ],
        "META": [
            r"\bon\s+instagram\b", r"\bposted\s+on\s+instagram\b", r"\binstagram\s+post\b",
            r"\binstagram\s+story\b", r"\binstagram\s+photo\b", r"\binstagram\s+account\b",
            r"\bfacebook\s+post\b", r"\bon\s+facebook\b", r"\bshared\s+on\s+facebook\b",
            r"\bwhatsapp\s+message\b", r"\bwhatsapp\s+chat\b", r"^\s*instagram\s*$",
            r"^\s*facebook\s*$", r"^\s*whatsapp\s*$",
            r"^\s*instagram(?:\s+instagram)?\s*$",
            r"^\s*facebook(?:\s+facebook)?\s*$",
            r"^\s*whatsapp(?:\s+whatsapp)?\s*$",
            r"\bfact\s+check\b", r"\bimage\b", r"\bphoto\b", r"\bselfie\b",
            r"\bcelebrit", r"\bactor\b", r"\bsinger\b", r"\bmodel\b",
            r"\bapologis", r"\bmom\b", r"\bboyfriend\b", r"\bgirlfriend\b",
            r"\bvacation\b", r"\bbikini\b", r"\bdating\b", r"\bfamily\b",
            r"\bholding\b", r"\bviral\s+image\b", r"\bphotos?\s+and\s+videos\b",
            r"\b@[a-z0-9_.]+\b", r"\byacht\b", r"\bsuperyacht\b",
            r"\bfor\s+whatsapp\b", r"\bextensions?\s+for\s+whatsapp\b"
        ],
        "TSLA": [
            r"\bwar\s+on\s+autopilot\b", r"\bbattlefield\s+robot\b", r"\budar\b",
            r"\bmilitary\s+robot\b", r"\bweapon(?:ized)?\s+robot\b",
            r"\bautopilot\b",
            r"\bfather\b", r"\bmother\b", r"\bfamily\b", r"\bchildhood\b",
            r"\berrol\s+musk\b", r"\bemerald\s+mine\b", r"\bgossip\b",
            r"\bthe\s+left\b", r"\bthe\s+right\b", r"\bliberal", r"\bconservative",
            r"\bpolitic", r"\bpolitical\b", r"\bcampaign\b", r"\bcommentator\b",
            r"\bscaramucci\b", r"\badviser\b", r"\badvisor\b",
            r"\brip\s+off\b", r"\bdoge\b"
        ],
    }

    CONTEXTUAL_KEEP_PATTERNS = {
        "AAPL": [
            r"\blaunch\b", r"\bunveil", r"\brelease\b", r"\bshipment", r"\bsales?\b",
            r"\bdemand\b", r"\brevenue\b", r"\bearnings\b", r"\bmarket\s+share\b",
            r"\bsupply\s+chain\b", r"\bproduction\b", r"\bpre-?order\b"
        ],
        "MSFT": [
            r"\bearnings\b", r"\brevenue\b", r"\bcommercial\b", r"\bcloud\b", r"\bsubscription\b",
            r"\brollout\b", r"\blaunch\b", r"\bfeature\b", r"\bpolicy\b", r"\bpartnership\b",
            r"\bacquisition\b", r"\bantitrust\b", r"\bsecurity\b", r"\boutage\b",
            r"\bgame\s+pass\b", r"\bmicrosoft\s+365\b", r"\boffice\s+365\b",
            r"\bphil\s+spencer\b", r"\bactivision\b", r"\bubisoft\b", r"\bazure\b"
        ],
        "GOOGL": [
            r"\bpolicy\b", r"\bmonetiz", r"\badvertis", r"\bsubscription\b", r"\bcreator\b",
            r"\bregulat", r"\bantitrust\b", r"\bearnings\b", r"\bfeature\b", r"\brollout\b",
            r"\bproduct\b", r"\bupdate\b"
        ],
        "META": [
            r"\bpolicy\b", r"\bmonetiz", r"\badvertis", r"\bfeature\b", r"\bupdate\b",
            r"\bregulat", r"\bban\b", r"\blaunch\b", r"\brollout\b", r"\bearnings\b",
            r"\bmoderat", r"\bcensorship\b", r"\bcontent\s+policy\b", r"\bfree\s+speech\b",
            r"\bonline\s+speech\b", r"\busers?\s+should\s+be\s+able\s+to\s+say\b",
            r"\bplatform\b", r"\boversight\b", r"\bboard\b",
            r"\bsection\s+230\b", r"\bhearing\b", r"\bceos?\b", r"\bdefend\b",
            r"\baccusations?\b", r"\bcongress\b", r"\bsenate\b"
        ],
        "TSLA": [
            r"\btesla\b", r"\bmodel\s+(s|3|x|y)\b", r"\bcybertruck\b", r"\bgigafactory\b",
            r"\bdeliver", r"\btesla\s+autopilot\b", r"\bfsd\b", r"\bsupercharger\b",
            r"\bmegapack\b", r"\btesla\s+energy\b", r"\bspacex\b",
            r"\bstarlink\b", r"\bxai\b", r"\btwitter\b", r"\bx\.com\b",
            r"\btesla\s+dashboard\b", r"\bdashboard\b", r"\bvehicle\b", r"\bdriver\b",
            r"\bsiriusxm\b", r"\bconnected\s+car\b", r"\blawsuit\b", r"\bsubpoena\b",
            r"\binvestor", r"\bstock\b", r"\bshares?\b", r"\bmarket\s+cap\b",
            r"\brecall\b", r"\bfactory\b", r"\brobotaxi\b", r"\bregulat",
            r"\bcrash\b", r"\bsafety\b", r"\binvestigat", r"\bnhtsa\b",
            r"\bdriver\s+monitoring\b", r"\bfatal\b", r"\bkilled\b", r"\bdead\b",
            r"\bacquisition\b", r"\btakeover\b"
        ],
        "NVDA": [
            r"\bnvidia\b", r"\bgeforce\b", r"\bcuda\b", r"\bomniverse\b", r"\bjetson\b",
            r"\brobotics\b", r"\bai\b", r"\bplatform\b", r"\baccelerat", r"\bpowered\s+by\b",
            r"\bwith\s+nvidia\b", r"\bh100\b", r"\bb200\b", r"\bdgx\b"
        ],
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
        base_use_slm = config.get('use_slm', True)
        enabled_symbols_raw = os.getenv("SLM_ENABLED_SYMBOLS", "*")
        disabled_symbols_raw = os.getenv("SLM_DISABLED_SYMBOLS", "")
        enabled_symbols = _parse_symbol_set(enabled_symbols_raw)
        disabled_symbols = _parse_symbol_set(disabled_symbols_raw)
        allow_all = enabled_symbols_raw.strip() == "*"
        self.use_slm = (
            base_use_slm
            and (allow_all or self.symbol.upper() in enabled_symbols)
            and self.symbol.upper() not in disabled_symbols
        )
        self.rule_verbose = os.getenv("RULE_VERBOSE", "false").lower() == "true"
        self.log_slm_interceptions = os.getenv("SLM_LOG_INTERCEPTIONS", "false").lower() == "true"

        # Pre-compile all patterns once at construction time to avoid per-call compilation overhead
        self._compiled_static_kill = [
            re.compile(p, re.IGNORECASE)
            for p in self.STATIC_KILL_PATTERNS.get(symbol, [])
        ]
        self._compiled_contextual_reject = [
            re.compile(p, re.IGNORECASE)
            for p in self.CONTEXTUAL_REJECT_PATTERNS.get(symbol, [])
        ]
        self._compiled_contextual_keep = [
            re.compile(p, re.IGNORECASE)
            for p in self.CONTEXTUAL_KEEP_PATTERNS.get(symbol, [])
        ]
        self._compiled_bypass_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self._ABSOLUTE_BYPASS_RAW.get(symbol, [])
        ]
        self._compiled_required_patterns = [
            (kw, re.compile(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', re.IGNORECASE))
            for kw in self.required_keywords
        ]
        too_common = self._TOO_COMMON_TO_BYPASS
        _primary_lower = {pk.lower() for pk in self.primary_keywords}
        strong_keywords = [
            k for k in self.expansion_keywords
            if len(k) > 3 and k.lower() not in _primary_lower and k.lower() not in too_common
        ]
        if strong_keywords:
            strong_keywords.sort(key=len, reverse=True)
            self._compiled_strong_pattern = re.compile(
                r'\b(' + '|'.join(re.escape(k) for k in strong_keywords) + r')\b',
                re.IGNORECASE,
            )
        else:
            self._compiled_strong_pattern = None
    
    def get_keywords(self, article_date):
        """返回必需的关键词"""
        return self.required_keywords

    @staticmethod
    def _keyword_matches(full_text, keyword):
        """Use word boundaries for all identity keywords to avoid substring false positives like intel -> intelnews."""
        return re.search(r'(?<!\w)' + re.escape(keyword) + r'(?!\w)', full_text, re.IGNORECASE)
    
    def should_include(self, article):
        title = article.get('title', '') or ''
        content = article.get('content', '') or ''
        # 回归稳健：不再将 URL 混入校验，防止 mu 在 /music/ 这种路径中误命中
        full_text = title + ' ' + content
        full_text_lower = full_text.lower()

        track_filter_step("entered")

        # 0. 防止空内容
        if not full_text.strip():
            return False

        # 2. 静态黑名单 (Static Kill)
        for _pat in self._compiled_static_kill:
            if _pat.search(full_text):
                track_filter_step("killed_by_static")
                if self.rule_verbose:
                    emit_rule_event(
                        "static_kill",
                        self.symbol,
                        f"🚫 Static Kill: {self.symbol} - Pattern: {_pat.pattern}",
                    )
                return False

        # 3. 关键词匹配计数 (必需关键词)
        matches = []
        for keyword, _pat in self._compiled_required_patterns:
            if _pat.search(full_text):
                matches.append(keyword)

        if len(matches) < self.min_matches:
            track_filter_step("killed_by_keyword")
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

            if not has_identity:
                track_filter_step("killed_by_strict")
                if self.rule_verbose:
                    emit_rule_event(
                        "strict_block",
                        self.symbol,
                        f"🛡️ Strict Identity BLOCK: {self.symbol} - Missing mandatory context. Title: {title[:60]}",
                    )
                return False
            else:
                if self.rule_verbose:
                    emit_rule_event(
                        "strict_pass",
                        self.symbol,
                        f"✅ Strict Identity PASS: {self.symbol} - Matched: {matched_identity} - Title: {title[:60]}",
                    )

        # 4.4 业务主体过滤：平台载体/商品页/外围品牌词不要直接因为品牌词而放行。
        if self._compiled_contextual_reject:
            hit_contextual_reject = any(_pat.search(full_text) for _pat in self._compiled_contextual_reject)
            if hit_contextual_reject:
                hit_contextual_keep = any(_pat.search(full_text) for _pat in self._compiled_contextual_keep)

                # TSLA + Starlink 只有在同时强烈讲 Tesla 业务本体时才保留。
                if self.symbol == "TSLA":
                    if not hit_contextual_keep:
                        track_filter_step("killed_by_contextual")
                        return False
                # GOOGL/META 平台类文章，只有在明确是平台业务/政策/产品更新时保留。
                elif self.symbol in {"GOOGL", "META"}:
                    if not hit_contextual_keep:
                        track_filter_step("killed_by_contextual")
                        return False
                # AAPL/AMZN 商品页、配件页直接拒绝；Apple 真正产品业务新闻靠 keep 词放行。
                else:
                    if not hit_contextual_keep:
                        track_filter_step("killed_by_contextual")
                        return False

        # 4.5 绝对强关联直通车 (Absolute Strong Bypass) — patterns pre-compiled in __init__
        for _pat in self._compiled_bypass_patterns:
            if _pat.search(full_text):
                track_filter_step("bypassed_by_absolute")
                if self.rule_verbose:
                    emit_rule_event(
                        "absolute_bypass",
                        self.symbol,
                        f"⚡ Absolute Bypass PASS: {self.symbol} - Pattern: {_pat.pattern} - Title: {title[:60]}",
                    )
                return True

        # 5. 强匹配逻辑：如果命中极其特殊的产品词，直接放行 Bypassing SLM — pattern pre-compiled in __init__
        if self._compiled_strong_pattern and self._compiled_strong_pattern.search(full_text):
            track_filter_step("bypassed_by_strong")
            return True

        # 5.5 Multiple identity hits are strong enough to pass without SLM.
        if len(matches) >= 2:
            track_filter_step("passed_by_multi_match")
            return True

        # 6. SLM 智能判断
        has_title = len(title.strip()) > 10
        # GDELT GKG 实体列数据（大量分号、无空格句子）不是真正的文章内容，不送 SLM
        _content_stripped = content.strip()
        _is_entity_data = (
            _content_stripped.count(";") > 3
            and len(_content_stripped.split()) < 20
        )
        has_content = len(_content_stripped) > 30 and not _content_stripped.startswith("http") and not _is_entity_data

        # Single weak identity hit should be verified semantically instead of default passing.
        if len(matches) == 1 and self.use_slm and (has_title or has_content):
            slm = get_slm_filter()
            if slm:
                is_relevant = slm.is_relevant(self.symbol, self.company_name, title, content)

                if not is_relevant:
                    track_filter_step("killed_by_slm")
                    if self.log_slm_interceptions:
                        log_dir = os.getenv(
                            "HISTORY_COLLECTOR_LOG_DIR",
                            os.path.expanduser("~/logs/history_collector"),
                        )
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, "slm_interceptions.log")

                        try:
                            with open(log_file, "a", encoding="utf-8") as f:
                                log_entry = (
                                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INTERCEPTED\n"
                                    f"Symbol: {self.symbol} | Title: {title}\n"
                                    f"Content: {content[:200]}...\n"
                                    f"{'-'*30}\n"
                                )
                                f.write(log_entry)
                        except:
                            pass

                    if self.rule_verbose:
                        emit_rule_event(
                            "slm_intercepted",
                            self.symbol,
                            f"🤖 SLM INTERCEPTED: {self.symbol} - Title: {title[:60]}",
                        )
                    return False
                else:
                    track_filter_step("slm_pass")
                    if self.rule_verbose:
                        emit_rule_event(
                            "slm_pass",
                            self.symbol,
                            f"✅ SLM PASS: {self.symbol} - Title: {title[:60]}",
                        )
                    return True

        if len(matches) == 1:
            track_filter_step("killed_single_match_without_slm")
            return False

        # 默认拒绝，避免 ambiguous 规则在弱信号下漏过。
        track_filter_step("killed_by_default")
        return False
