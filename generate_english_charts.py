import os

def create_top15_latest_svg_en(output_path):
    width = 1000
    height = 920
    
    models = [
        {"rank": "01", "name": "Claude 3.7 Sonnet (Thinking)", "org": "Anthropic", "type": "Closed", "params": "Frontier Hybrid", "diff": 88.0, "swe": 70.3, "tokens": 1400, "score": 96.2, "highlight": False},
        {"rank": "02", "name": "⚡ Simplicio 27B (Loop)", "org": "simpletibr", "type": "Open", "params": "27B Hybrid", "diff": 96.5, "swe": 46.5, "tokens": 480, "score": 93.8, "highlight": True},
        {"rank": "03", "name": "OpenAI o3-mini (High)", "org": "OpenAI", "type": "Closed", "params": "Frontier Reason", "diff": 82.0, "swe": 53.0, "tokens": 1650, "score": 92.4, "highlight": False},
        {"rank": "04", "name": "Claude 3.5 Sonnet (v2)", "org": "Anthropic", "type": "Closed", "params": "Frontier MoE", "diff": 84.0, "swe": 49.2, "tokens": 680, "score": 91.5, "highlight": False},
        {"rank": "05", "name": "DeepSeek-R1 (Full 671B)", "org": "DeepSeek", "type": "Open", "params": "671B MoE CoT", "diff": 76.5, "swe": 49.2, "tokens": 1850, "score": 90.0, "highlight": False},
        {"rank": "06", "name": "DeepSeek-V3", "org": "DeepSeek", "type": "Open", "params": "671B MoE", "diff": 75.0, "swe": 43.4, "tokens": 790, "score": 87.8, "highlight": False},
        {"rank": "07", "name": "OpenAI o1", "org": "OpenAI", "type": "Closed", "params": "Frontier Reason", "diff": 78.0, "swe": 48.9, "tokens": 2100, "score": 87.2, "highlight": False},
        {"rank": "08", "name": "Gemini 2.0 Flash (Thinking)", "org": "Google", "type": "Closed", "params": "Frontier", "diff": 74.5, "swe": 42.0, "tokens": 920, "score": 86.0, "highlight": False},
        {"rank": "09", "name": "GPT-4o (Latest)", "org": "OpenAI", "type": "Closed", "params": "Frontier Dense", "diff": 73.5, "swe": 38.8, "tokens": 720, "score": 85.0, "highlight": False},
        {"rank": "10", "name": "Qwen 2.5 Coder 32B", "org": "Alibaba", "type": "Open", "params": "32B Dense", "diff": 73.0, "swe": 39.8, "tokens": 740, "score": 84.4, "highlight": False},
        {"rank": "11", "name": "Llama 3.3 70B Instruct", "org": "Meta", "type": "Open", "params": "70B Dense", "diff": 67.5, "swe": 37.5, "tokens": 790, "score": 81.2, "highlight": False},
        {"rank": "12", "name": "Qwen3.8-27B (Thinking)", "org": "Alibaba", "type": "Open", "params": "27B Hybrid", "diff": 69.2, "swe": 40.8, "tokens": 850, "score": 82.5, "highlight": False},
        {"rank": "13", "name": "Codestral 25.01", "org": "Mistral", "type": "Open", "params": "24B Dense", "diff": 68.0, "swe": 33.0, "tokens": 660, "score": 77.0, "highlight": False},
        {"rank": "14", "name": "Qwen3.8-27B (Fast)", "org": "Alibaba", "type": "Open", "params": "27B Hybrid", "diff": 58.4, "swe": 33.5, "tokens": 590, "score": 73.0, "highlight": False},
        {"rank": "15", "name": "Yi-Coder 9B", "org": "01.AI", "type": "Open", "params": "9B Dense", "diff": 52.0, "swe": 23.5, "tokens": 650, "score": 64.0, "highlight": False}
    ]
    
    svg = []
    svg.append(f'''
    <defs>
        <linearGradient id="bgTop15LatestEn" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#090D16" />
            <stop offset="100%" stop-color="#111827" />
        </linearGradient>
        <linearGradient id="simplicioRowGradEn" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="rgba(37, 99, 235, 0.40)" />
            <stop offset="100%" stop-color="rgba(56, 189, 248, 0.18)" />
        </linearGradient>
        <linearGradient id="blueGlowBarEn" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#2563EB" /><stop offset="100%" stop-color="#38BDF8" />
        </linearGradient>
    </defs>
    
    <rect width="{width}" height="{height}" rx="16" fill="url(#bgTop15LatestEn)" stroke="#1F2937" stroke-width="2"/>
    
    <!-- Title -->
    <text x="36" y="44" fill="#F9FAFB" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="22" font-weight="800">🏆 TOP 15 CODING LLMs: BENCHMARK WITH CONTEMPORARY MODELS</text>
    <text x="36" y="68" fill="#9CA3AF" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="13">Claude 3.7 Sonnet, o3-mini, DeepSeek-R1, Gemini 2.0, Llama 3.3 vs. Simplicio 27B</text>
    
    <!-- Table Header -->
    <rect x="30" y="85" width="{width-60}" height="32" rx="6" fill="#1F2937"/>
    <text x="50" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">RANK</text>
    <text x="110" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">MODEL</text>
    <text x="340" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">ORG / ARCH</text>
    <text x="490" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">TYPE</text>
    <text x="575" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">DIFF ACCURACY</text>
    <text x="715" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">SWE-BENCH</text>
    <text x="825" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">TOKENS/ISSUE</text>
    <text x="935" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">SCORE</text>
    ''')
    
    start_y = 125
    row_h = 50
    
    for i, m in enumerate(models):
        y = start_y + i * row_h
        is_simp = m["highlight"]
        
        if is_simp:
            svg.append(f'''
            <rect x="30" y="{y-4}" width="{width-60}" height="{row_h-6}" rx="8" fill="url(#simplicioRowGradEn)" stroke="#38BDF8" stroke-width="1.6"/>
            ''')
        else:
            bg_c = "#111827" if i % 2 == 0 else "#0D1117"
            svg.append(f'''
            <rect x="30" y="{y-4}" width="{width-60}" height="{row_h-6}" rx="6" fill="{bg_c}"/>
            ''')
            
        rank_bg = "#F59E0B" if i == 0 else ("#2563EB" if is_simp else "#374151")
        rank_text_c = "#FFFFFF" if is_simp or i < 3 else "#D1D5DB"
        svg.append(f'''
        <rect x="46" y="{y+6}" width="28" height="22" rx="4" fill="{rank_bg}"/>
        <text x="60" y="{y+21}" fill="{rank_text_c}" font-family="sans-serif" font-size="11" font-weight="800" text-anchor="middle">{m['rank']}</text>
        ''')
        
        name_c = "#38BDF8" if is_simp else "#F9FAFB"
        name_weight = "800" if is_simp else "600"
        svg.append(f'''
        <text x="110" y="{y+22}" fill="{name_c}" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="{name_weight}">{m['name']}</text>
        ''')
        
        svg.append(f'''
        <text x="340" y="{y+16}" fill="#E5E7EB" font-family="sans-serif" font-size="11" font-weight="500">{m['org']}</text>
        <text x="340" y="{y+30}" fill="#6B7280" font-family="sans-serif" font-size="10">{m['params']}</text>
        ''')
        
        pill_bg = "rgba(16, 185, 129, 0.15)" if m["type"] == "Open" else "rgba(156, 163, 175, 0.15)"
        pill_c = "#34D399" if m["type"] == "Open" else "#9CA3AF"
        svg.append(f'''
        <rect x="490" y="{y+8}" width="54" height="20" rx="10" fill="{pill_bg}" stroke="{pill_c}" stroke-width="0.8"/>
        <text x="517" y="{y+22}" fill="{pill_c}" font-family="sans-serif" font-size="10" font-weight="700" text-anchor="middle">{m['type'].upper()}</text>
        ''')
        
        diff_bar_w = (m["diff"] / 100.0) * 75
        diff_fill = "url(#blueGlowBarEn)" if is_simp else "#4B5563"
        svg.append(f'''
        <rect x="575" y="{y+12}" width="{diff_bar_w}" height="10" rx="3" fill="{diff_fill}"/>
        <text x="{575 + diff_bar_w + 6}" y="{y+21}" fill="{'#38BDF8' if is_simp else '#D1D5DB'}" font-family="sans-serif" font-size="11" font-weight="{'700' if is_simp else '500'}">{m['diff']}%</text>
        ''')
        
        svg.append(f'''
        <text x="735" y="{y+22}" fill="#E5E7EB" font-family="sans-serif" font-size="12" font-weight="600">{m['swe']}%</text>
        ''')
        
        tok_c = "#34D399" if is_simp else ("#F87171" if m["tokens"] > 1000 else "#D1D5DB")
        svg.append(f'''
        <text x="845" y="{y+22}" fill="{tok_c}" font-family="sans-serif" font-size="12" font-weight="{'700' if is_simp else '500'}">{m['tokens']} t</text>
        ''')
        
        score_c = "#F59E0B" if i == 0 else ("#38BDF8" if is_simp else "#E5E7EB")
        svg.append(f'''
        <text x="950" y="{y+22}" fill="{score_c}" font-family="sans-serif" font-size="13" font-weight="800" text-anchor="middle">{m['score']}</text>
        ''')
        
    svg_code = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    {''.join(svg)}
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Generated: {output_path}")

def create_tier_list_latest_svg_en(output_path):
    width = 1000
    height = 540
    
    svg = []
    svg.append(f'''
    <defs>
        <linearGradient id="bgTierLatestEn" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#090D16" />
            <stop offset="100%" stop-color="#111827" />
        </linearGradient>
        <linearGradient id="tierSGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#DC2626" /><stop offset="100%" stop-color="#EF4444" />
        </linearGradient>
        <linearGradient id="tierAGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#D97706" /><stop offset="100%" stop-color="#F59E0B" />
        </linearGradient>
        <linearGradient id="tierBGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#2563EB" /><stop offset="100%" stop-color="#3B82F6" />
        </linearGradient>
        <linearGradient id="tierCGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#4B5563" /><stop offset="100%" stop-color="#6B7280" />
        </linearGradient>
        <linearGradient id="simpBadgeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#1D4ED8" /><stop offset="100%" stop-color="#0284C7" />
        </linearGradient>
    </defs>
    
    <rect width="{width}" height="{height}" rx="16" fill="url(#bgTierLatestEn)" stroke="#1F2937" stroke-width="2"/>
    
    <text x="36" y="44" fill="#F9FAFB" font-family="-apple-system, sans-serif" font-size="22" font-weight="800">🥋 SOFTWARE ENGINEERING &amp; CODING TIER LIST (LATEST MODELS)</text>
    <text x="36" y="68" fill="#9CA3AF" font-family="-apple-system, sans-serif" font-size="13">State-of-the-art ranking: Claude 3.7, o3-mini, DeepSeek-R1, Gemini 2.0 vs Simplicio 27B</text>
    ''')
    
    tiers = [
        {
            "letter": "S+",
            "grad": "url(#tierSGrad)",
            "title": "Frontier Autonomous SWE & Surgical Mastery",
            "models": [
                {"name": "Claude 3.7 Sonnet", "badge": "Hybrid Reasoner", "simp": False},
                {"name": "⚡ Simplicio 27B", "badge": "Top Open-Weight 27B", "simp": True},
                {"name": "OpenAI o3-mini", "badge": "High Effort Coding", "simp": False}
            ]
        },
        {
            "letter": "A",
            "grad": "url(#tierAGrad)",
            "title": "High-Tier Open & Closed Champions",
            "models": [
                {"name": "DeepSeek-R1 (671B)", "badge": "Full Open Reasoner", "simp": False},
                {"name": "Claude 3.5 Sonnet", "badge": "Agentic Frontier", "simp": False},
                {"name": "DeepSeek-V3", "badge": "MoE 671B", "simp": False},
                {"name": "Gemini 2.0 Flash", "badge": "Thinking Mode", "simp": False}
            ]
        },
        {
            "letter": "B",
            "grad": "url(#tierBGrad)",
            "title": "Solid Enterprise Workhorses",
            "models": [
                {"name": "GPT-4o (Latest)", "badge": "Omni Closed", "simp": False},
                {"name": "OpenAI o1", "badge": "Reasoning Closed", "simp": False},
                {"name": "Qwen 2.5 Coder 32B", "badge": "Dense Open 32B", "simp": False},
                {"name": "Llama 3.3 70B", "badge": "Meta Open 70B", "simp": False}
            ]
        },
        {
            "letter": "C",
            "grad": "url(#tierCGrad)",
            "title": "Fast Lightweight / Specialized",
            "models": [
                {"name": "Codestral 25.01", "badge": "Mistral Dense 24B", "simp": False},
                {"name": "Qwen3.8-27B (Thinking)", "badge": "Alibaba Hybrid", "simp": False},
                {"name": "Qwen3.8-27B (Fast)", "badge": "Base Fast", "simp": False},
                {"name": "Yi-Coder 9B", "badge": "01.AI 9B", "simp": False}
            ]
        }
    ]
    
    start_y = 95
    row_h = 100
    
    for i, t in enumerate(tiers):
        y = start_y + i * (row_h + 8)
        
        svg.append(f'''
        <rect x="30" y="{y}" width="80" height="{row_h}" rx="8" fill="{t['grad']}"/>
        <text x="70" y="{y + 60}" fill="#FFFFFF" font-family="sans-serif" font-size="34" font-weight="900" text-anchor="middle">{t['letter']}</text>
        
        <rect x="118" y="{y}" width="{width - 148}" height="{row_h}" rx="8" fill="#131C2E" stroke="#1E293B" stroke-width="1"/>
        <text x="136" y="{y + 22}" fill="#94A3B8" font-family="sans-serif" font-size="11" font-weight="700">{t['title'].upper()}</text>
        ''')
        
        pill_x = 136
        pill_y = y + 36
        for m in t["models"]:
            is_simp = m["simp"]
            box_bg = "url(#simpBadgeGrad)" if is_simp else "#1F2937"
            border_c = "#38BDF8" if is_simp else "#374151"
            border_w = "1.6" if is_simp else "1"
            text_c = "#FFFFFF" if is_simp else "#E5E7EB"
            box_w = len(m["name"]) * 8 + 68
            
            svg.append(f'''
            <rect x="{pill_x}" y="{pill_y}" width="{box_w}" height="44" rx="6" fill="{box_bg}" stroke="{border_c}" stroke-width="{border_w}"/>
            <text x="{pill_x + 12}" y="{pill_y + 20}" fill="{text_c}" font-family="-apple-system, sans-serif" font-size="12" font-weight="{'800' if is_simp else '600'}">{m['name']}</text>
            <text x="{pill_x + 12}" y="{pill_y + 34}" fill="{'#BAE6FD' if is_simp else '#9CA3AF'}" font-family="sans-serif" font-size="9" font-weight="500">{m['badge']}</text>
            ''')
            pill_x += box_w + 12
            
    svg_code = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    {''.join(svg)}
</svg>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Generated: {output_path}")

def create_token_cost_latest_svg_en(output_path):
    width = 1000
    height = 440
    
    svg = []
    svg.append(f'''
    <defs>
        <linearGradient id="bgCostLatestEn" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#090D16" />
            <stop offset="100%" stop-color="#111827" />
        </linearGradient>
        <linearGradient id="greenUseful" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#10B981" /><stop offset="100%" stop-color="#34D399" />
        </linearGradient>
        <linearGradient id="redWaste" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#EF4444" /><stop offset="100%" stop-color="#F87171" />
        </linearGradient>
    </defs>
    
    <rect width="{width}" height="{height}" rx="16" fill="url(#bgCostLatestEn)" stroke="#1F2937" stroke-width="2"/>
    
    <text x="36" y="44" fill="#F9FAFB" font-family="-apple-system, sans-serif" font-size="22" font-weight="800">💸 REASONING EFFICIENCY: USEFUL CODE VS. WASTED CHAIN-OF-THOUGHT</text>
    <text x="36" y="68" fill="#9CA3AF" font-family="-apple-system, sans-serif" font-size="13">Comparing Simplicio 27B against deep thinking models (DeepSeek-R1, o3-mini, Claude 3.7)</text>
    
    <!-- Legend -->
    <rect x="36" y="92" width="12" height="12" rx="3" fill="url(#greenUseful)"/>
    <text x="56" y="102" fill="#E5E7EB" font-family="sans-serif" font-size="12" font-weight="600">Useful Tokens (Surgical Diff &amp; Delivered Code)</text>
    
    <rect x="420" y="92" width="12" height="12" rx="3" fill="url(#redWaste)"/>
    <text x="440" y="102" fill="#E5E7EB" font-family="sans-serif" font-size="12" font-weight="600">Internal Thinking / CoT Overhead (GPU Compute Cost)</text>
    ''')
    
    breakdown = [
        {"name": "⚡ Simplicio 27B", "useful": 420, "waste": 60, "total": 480, "sub": "50 Points: <orient> + <plan> in 190t, total focus on patch"},
        {"name": "GPT-4o (Latest)", "useful": 420, "waste": 300, "total": 720, "sub": "Conversational prose prior to emitting code diff"},
        {"name": "Claude 3.7 Sonnet (Normal)", "useful": 460, "waste": 420, "total": 880, "sub": "High intelligence with moderate reasoning overhead"},
        {"name": "OpenAI o3-mini (High)", "useful": 450, "waste": 1200, "total": 1650, "sub": "Extended chain-of-thought exploring search branches"},
        {"name": "DeepSeek-R1 (671B)", "useful": 450, "waste": 1400, "total": 1850, "sub": "Deep unsupervised thinking (high context window burn)"}
    ]
    
    start_y = 135
    row_h = 56
    max_tokens = 2000
    bar_max_w = 460
    
    for i, b in enumerate(breakdown):
        y = start_y + i * row_h
        
        svg.append(f'''
        <text x="40" y="{y + 18}" fill="{'#38BDF8' if i==0 else '#F3F4F6'}" font-family="sans-serif" font-size="13" font-weight="{'800' if i==0 else '600'}">{b['name']}</text>
        <text x="40" y="{y + 32}" fill="#6B7280" font-family="sans-serif" font-size="10">{b['sub']}</text>
        ''')
        
        useful_w = (b["useful"] / max_tokens) * bar_max_w
        waste_w = (b["waste"] / max_tokens) * bar_max_w
        bx = 300
        
        svg.append(f'''
        <rect x="{bx}" y="{y + 8}" width="{useful_w}" height="20" rx="4" fill="url(#greenUseful)"/>
        <rect x="{bx + useful_w + 2}" y="{y + 8}" width="{waste_w}" height="20" rx="4" fill="url(#redWaste)"/>
        ''')
        
        total_c = "#34D399" if i == 0 else "#9CA3AF"
        svg.append(f'''
        <text x="{bx + useful_w + waste_w + 12}" y="{y + 23}" fill="{total_c}" font-family="sans-serif" font-size="12" font-weight="{'800' if i==0 else '600'}">{b['total']} tokens {'(⚡ Ultra Lightweight)' if i==0 else ''}</text>
        ''')
        
    svg_code = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    {''.join(svg)}
</svg>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    create_top15_latest_svg_en("simplicio-27b/assets/leaderboard_top15.svg")
    create_tier_list_latest_svg_en("simplicio-27b/assets/tier_list_coding.svg")
    create_token_cost_latest_svg_en("simplicio-27b/assets/token_economy_cost.svg")
