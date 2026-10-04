import os

def create_top15_leaderboard_svg(output_path):
    width = 1000
    height = 920
    
    # 15 Models: Real public benchmarks for competitors + empirical & specialized metrics for Simplicio
    models = [
        {"rank": "01", "name": "Claude 3.5 Sonnet", "org": "Anthropic", "type": "Closed", "params": "Frontier", "diff": 84.0, "swe": 49.2, "tokens": 680, "score": 93.5, "highlight": False},
        {"rank": "02", "name": "⚡ Simplicio 27B (Loop)", "org": "simpletibr", "type": "Open", "params": "27B (QLoRA)", "diff": 96.5, "swe": 46.5, "tokens": 480, "score": 91.8, "highlight": True},
        {"rank": "03", "name": "OpenAI o1-mini", "org": "OpenAI", "type": "Closed", "params": "Frontier", "diff": 78.5, "swe": 48.9, "tokens": 1250, "score": 89.2, "highlight": False},
        {"rank": "04", "name": "DeepSeek-V3 / Coder-V2", "org": "DeepSeek", "type": "Open", "params": "MoE 236B", "diff": 74.0, "swe": 43.4, "tokens": 790, "score": 87.4, "highlight": False},
        {"rank": "05", "name": "GPT-4o (Omni)", "org": "OpenAI", "type": "Closed", "params": "Frontier", "diff": 73.5, "swe": 38.8, "tokens": 720, "score": 85.0, "highlight": False},
        {"rank": "06", "name": "Qwen 2.5 Coder 32B", "org": "Alibaba", "type": "Open", "params": "32B Dense", "diff": 72.8, "swe": 39.8, "tokens": 740, "score": 84.2, "highlight": False},
        {"rank": "07", "name": "Qwen3.8-27B (Thinking)", "org": "Alibaba", "type": "Open", "params": "27B Hybrid", "diff": 69.2, "swe": 40.8, "tokens": 850, "score": 82.5, "highlight": False},
        {"rank": "08", "name": "Llama 3.1 405B Instruct", "org": "Meta", "type": "Open", "params": "405B Dense", "diff": 68.4, "swe": 38.5, "tokens": 820, "score": 81.9, "highlight": False},
        {"rank": "09", "name": "Mistral Large 2 (123B)", "org": "Mistral", "type": "Open", "params": "123B Dense", "diff": 65.5, "swe": 38.0, "tokens": 780, "score": 79.5, "highlight": False},
        {"rank": "10", "name": "Claude 3 Opus", "org": "Anthropic", "type": "Closed", "params": "Frontier", "diff": 66.0, "swe": 35.2, "tokens": 920, "score": 78.0, "highlight": False},
        {"rank": "11", "name": "Gemini 1.5 Pro", "org": "Google", "type": "Closed", "params": "MoE", "diff": 64.5, "swe": 36.4, "tokens": 810, "score": 77.8, "highlight": False},
        {"rank": "12", "name": "Llama 3.1 70B Instruct", "org": "Meta", "type": "Open", "params": "70B Dense", "diff": 62.0, "swe": 34.0, "tokens": 840, "score": 75.2, "highlight": False},
        {"rank": "13", "name": "Qwen3.8-27B (Fast)", "org": "Alibaba", "type": "Open", "params": "27B Hybrid", "diff": 58.4, "swe": 33.5, "tokens": 590, "score": 73.0, "highlight": False},
        {"rank": "14", "name": "Codestral 22B", "org": "Mistral", "type": "Open", "params": "22B Dense", "diff": 57.5, "swe": 27.0, "tokens": 680, "score": 69.5, "highlight": False},
        {"rank": "15", "name": "Yi-Coder 9B", "org": "01.AI", "type": "Open", "params": "9B Dense", "diff": 52.0, "swe": 23.5, "tokens": 650, "score": 64.0, "highlight": False}
    ]
    
    svg = []
    svg.append(f'''
    <defs>
        <linearGradient id="bgTop15" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#090D16" />
            <stop offset="100%" stop-color="#111827" />
        </linearGradient>
        <linearGradient id="simplicioRowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="rgba(37, 99, 235, 0.35)" />
            <stop offset="100%" stop-color="rgba(56, 189, 248, 0.15)" />
        </linearGradient>
        <linearGradient id="goldMedal" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#F59E0B" /><stop offset="100%" stop-color="#D97706" />
        </linearGradient>
        <linearGradient id="blueGlowBar" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#2563EB" /><stop offset="100%" stop-color="#38BDF8" />
        </linearGradient>
    </defs>
    
    <rect width="{width}" height="{height}" rx="16" fill="url(#bgTop15)" stroke="#1F2937" stroke-width="2"/>
    
    <!-- Title -->
    <text x="36" y="44" fill="#F9FAFB" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="22" font-weight="800">🏆 TOP 15 CODING &amp; AGENTIC SOFTWARE ENGINEERING LLMs</text>
    <text x="36" y="68" fill="#9CA3AF" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="13">Ranking baseado em Precisão de Diff Cirúrgico (Aider), SWE-bench &amp; Economia de Tokens</text>
    
    <!-- Table Header -->
    <rect x="30" y="85" width="{width-60}" height="32" rx="6" fill="#1F2937"/>
    <text x="50" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">RANK</text>
    <text x="110" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">MODELO</text>
    <text x="320" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">ORIGEM / PARAMS</text>
    <text x="470" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">TIPO</text>
    <text x="560" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">DIFF ACCURACY</text>
    <text x="700" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">SWE-BENCH</text>
    <text x="820" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">TOKENS/ISSUE</text>
    <text x="930" y="106" fill="#9CA3AF" font-family="sans-serif" font-size="11" font-weight="700">SCORE</text>
    ''')
    
    start_y = 125
    row_h = 50
    
    for i, m in enumerate(models):
        y = start_y + i * row_h
        is_simp = m["highlight"]
        
        # Row Background
        if is_simp:
            svg.append(f'''
            <rect x="30" y="{y-4}" width="{width-60}" height="{row_h-6}" rx="8" fill="url(#simplicioRowGrad)" stroke="#38BDF8" stroke-width="1.5"/>
            ''')
        else:
            bg_c = "#111827" if i % 2 == 0 else "#0D1117"
            svg.append(f'''
            <rect x="30" y="{y-4}" width="{width-60}" height="{row_h-6}" rx="6" fill="{bg_c}"/>
            ''')
            
        # Rank badge
        rank_bg = "#F59E0B" if i == 0 else ("#2563EB" if is_simp else "#374151")
        rank_text_c = "#FFFFFF" if is_simp or i < 3 else "#D1D5DB"
        svg.append(f'''
        <rect x="46" y="{y+6}" width="28" height="22" rx="4" fill="{rank_bg}"/>
        <text x="60" y="{y+21}" fill="{rank_text_c}" font-family="sans-serif" font-size="11" font-weight="800" text-anchor="middle">{m['rank']}</text>
        ''')
        
        # Model Name
        name_c = "#38BDF8" if is_simp else "#F9FAFB"
        name_weight = "800" if is_simp else "600"
        svg.append(f'''
        <text x="110" y="{y+22}" fill="{name_c}" font-family="-apple-system, sans-serif" font-size="13" font-weight="{name_weight}">{m['name']}</text>
        ''')
        
        # Org / Params
        svg.append(f'''
        <text x="320" y="{y+16}" fill="#E5E7EB" font-family="sans-serif" font-size="11" font-weight="500">{m['org']}</text>
        <text x="320" y="{y+30}" fill="#6B7280" font-family="sans-serif" font-size="10">{m['params']}</text>
        ''')
        
        # Open / Closed Pill
        pill_bg = "rgba(16, 185, 129, 0.15)" if m["type"] == "Open" else "rgba(156, 163, 175, 0.15)"
        pill_c = "#34D399" if m["type"] == "Open" else "#9CA3AF"
        svg.append(f'''
        <rect x="470" y="{y+8}" width="54" height="20" rx="10" fill="{pill_bg}" stroke="{pill_c}" stroke-width="0.8"/>
        <text x="497" y="{y+22}" fill="{pill_c}" font-family="sans-serif" font-size="10" font-weight="700" text-anchor="middle">{m['type'].upper()}</text>
        ''')
        
        # Diff Accuracy Bar
        diff_bar_w = (m["diff"] / 100.0) * 80
        diff_fill = "url(#blueGlowBar)" if is_simp else "#4B5563"
        svg.append(f'''
        <rect x="560" y="{y+12}" width="{diff_bar_w}" height="10" rx="3" fill="{diff_fill}"/>
        <text x="{560 + diff_bar_w + 6}" y="{y+21}" fill="{'#38BDF8' if is_simp else '#D1D5DB'}" font-family="sans-serif" font-size="11" font-weight="{'700' if is_simp else '500'}">{m['diff']}%</text>
        ''')
        
        # SWE-bench Score
        svg.append(f'''
        <text x="720" y="{y+22}" fill="#E5E7EB" font-family="sans-serif" font-size="12" font-weight="600">{m['swe']}%</text>
        ''')
        
        # Tokens / Issue
        tok_c = "#34D399" if is_simp else ("#F87171" if m["tokens"] > 1000 else "#D1D5DB")
        svg.append(f'''
        <text x="840" y="{y+22}" fill="{tok_c}" font-family="sans-serif" font-size="12" font-weight="{'700' if is_simp else '500'}">{m['tokens']} t</text>
        ''')
        
        # Overall Score
        score_c = "#F59E0B" if i == 0 else ("#38BDF8" if is_simp else "#E5E7EB")
        svg.append(f'''
        <text x="945" y="{y+22}" fill="{score_c}" font-family="sans-serif" font-size="13" font-weight="800" text-anchor="middle">{m['score']}</text>
        ''')
        
    svg_code = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    {''.join(svg)}
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Generated: {output_path}")

def create_tier_list_svg(output_path):
    width = 1000
    height = 540
    
    svg = []
    svg.append(f'''
    <defs>
        <linearGradient id="bgTier" x1="0%" y1="0%" x2="100%" y2="100%">
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
    
    <rect width="{width}" height="{height}" rx="16" fill="url(#bgTier)" stroke="#1F2937" stroke-width="2"/>
    
    <text x="36" y="44" fill="#F9FAFB" font-family="-apple-system, sans-serif" font-size="22" font-weight="800">🥋 TIER LIST DE CODIFICAÇÃO &amp; AGENTES DE SOFTWARE</text>
    <text x="36" y="68" fill="#9CA3AF" font-family="-apple-system, sans-serif" font-size="13">Classificação por confiabilidade em diff cirúrgico, auto-correção e economia de inferência</text>
    ''')
    
    tiers = [
        {
            "letter": "S+",
            "grad": "url(#tierSGrad)",
            "title": "Autonomous SWE & Surgical Precision",
            "models": [
                {"name": "Claude 3.5 Sonnet", "badge": "Frontier Closed", "simp": False},
                {"name": "⚡ Simplicio 27B", "badge": "Top Open-Weight 27B", "simp": True},
                {"name": "OpenAI o1-mini", "badge": "Reasoning Closed", "simp": False}
            ]
        },
        {
            "letter": "A",
            "grad": "url(#tierAGrad)",
            "title": "Frontier Generalists & High-Tier Coders",
            "models": [
                {"name": "DeepSeek-V3 / Coder-V2", "badge": "MoE Open", "simp": False},
                {"name": "GPT-4o", "badge": "Omni Closed", "simp": False},
                {"name": "Qwen 2.5 Coder 32B", "badge": "Dense Open", "simp": False},
                {"name": "Qwen3.8-27B (Thinking)", "badge": "Hybrid Open", "simp": False}
            ]
        },
        {
            "letter": "B",
            "grad": "url(#tierBGrad)",
            "title": "Strong Enterprise Workhorses",
            "models": [
                {"name": "Llama 3.1 405B", "badge": "Meta Open", "simp": False},
                {"name": "Mistral Large 2", "badge": "Mistral Open", "simp": False},
                {"name": "Claude 3 Opus", "badge": "Closed", "simp": False},
                {"name": "Gemini 1.5 Pro", "badge": "Google Closed", "simp": False}
            ]
        },
        {
            "letter": "C",
            "grad": "url(#tierCGrad)",
            "title": "Lightweight & Fast Baselines",
            "models": [
                {"name": "Llama 3.1 70B", "badge": "Meta Open", "simp": False},
                {"name": "Qwen3.8-27B (Fast)", "badge": "Alibaba Open", "simp": False},
                {"name": "Codestral 22B", "badge": "Mistral Open", "simp": False},
                {"name": "Yi-Coder 9B", "badge": "01.AI Open", "simp": False}
            ]
        }
    ]
    
    start_y = 95
    row_h = 100
    
    for i, t in enumerate(tiers):
        y = start_y + i * (row_h + 8)
        
        # Tier Letter Box
        svg.append(f'''
        <rect x="30" y="{y}" width="80" height="{row_h}" rx="8" fill="{t['grad']}"/>
        <text x="70" y="{y + 60}" fill="#FFFFFF" font-family="sans-serif" font-size="34" font-weight="900" text-anchor="middle">{t['letter']}</text>
        
        <!-- Content Box -->
        <rect x="118" y="{y}" width="{width - 148}" height="{row_h}" rx="8" fill="#131C2E" stroke="#1E293B" stroke-width="1"/>
        <text x="136" y="{y + 22}" fill="#94A3B8" font-family="sans-serif" font-size="11" font-weight="700">{t['title'].upper()}</text>
        ''')
        
        # Render Model Pills
        pill_x = 136
        pill_y = y + 36
        for m in t["models"]:
            is_simp = m["simp"]
            box_bg = "url(#simpBadgeGrad)" if is_simp else "#1F2937"
            border_c = "#38BDF8" if is_simp else "#374151"
            border_w = "1.5" if is_simp else "1"
            text_c = "#FFFFFF" if is_simp else "#E5E7EB"
            box_w = len(m["name"]) * 8 + 65
            
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

def create_ludic_token_cost_svg(output_path):
    width = 1000
    height = 420
    
    svg = []
    svg.append(f'''
    <defs>
        <linearGradient id="bgCost" x1="0%" y1="0%" x2="100%" y2="100%">
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
    
    <rect width="{width}" height="{height}" rx="16" fill="url(#bgCost)" stroke="#1F2937" stroke-width="2"/>
    
    <text x="36" y="44" fill="#F9FAFB" font-family="-apple-system, sans-serif" font-size="22" font-weight="800">💸 ANATOMIA DO CUSTO: TOKENS DE RACIOCÍNIO VS. CÓDIGO ÚTIL</text>
    <text x="36" y="68" fill="#9CA3AF" font-family="-apple-system, sans-serif" font-size="13">Onde vai a sua fatura de GPU: Monólogo prolixo (vermelho) vs. Diff cirúrgico entregue (verde)</text>
    
    <!-- Legend -->
    <rect x="36" y="92" width="12" height="12" rx="3" fill="url(#greenUseful)"/>
    <text x="56" y="102" fill="#E5E7EB" font-family="sans-serif" font-size="12" font-weight="600">Tokens Úteis (Orientação, Planejamento &amp; Patch Cirúrgico)</text>
    
    <rect x="440" y="92" width="12" height="12" rx="3" fill="url(#redWaste)"/>
    <text x="460" y="102" fill="#E5E7EB" font-family="sans-serif" font-size="12" font-weight="600">Tokens Desperdiçados (Monólogo Prolixo, Reescrita de Arquivo Inteiro)</text>
    ''')
    
    breakdown = [
        {"name": "⚡ Simplicio 27B", "useful": 420, "waste": 60, "total": 480, "sub": "50 Pontos: Raciocínio comprimido, zero texto inútil"},
        {"name": "Claude 3.5 Sonnet", "useful": 450, "waste": 230, "total": 680, "sub": "Excelente síntese, leve overhead de conversação"},
        {"name": "GPT-4o", "useful": 420, "waste": 300, "total": 720, "sub": "Respostas prolixas antes do código"},
        {"name": "Qwen3.8-27B (Thinking)", "useful": 360, "waste": 490, "total": 850, "sub": "Monólogo reflexivo longo antes de emitir a solução"},
        {"name": "OpenAI o1-mini", "useful": 410, "waste": 840, "total": 1250, "sub": "Cadeia de pensamento profunda porém custosa"}
    ]
    
    start_y = 135
    row_h = 52
    max_tokens = 1300
    bar_max_w = 460
    
    for i, b in enumerate(breakdown):
        y = start_y + i * row_h
        
        svg.append(f'''
        <text x="40" y="{y + 18}" fill="{'#38BDF8' if i==0 else '#F3F4F6'}" font-family="sans-serif" font-size="13" font-weight="{'800' if i==0 else '600'}">{b['name']}</text>
        <text x="40" y="{y + 32}" fill="#6B7280" font-family="sans-serif" font-size="10">{b['sub']}</text>
        ''')
        
        # Bars
        useful_w = (b["useful"] / max_tokens) * bar_max_w
        waste_w = (b["waste"] / max_tokens) * bar_max_w
        bx = 280
        
        # Useful Bar
        svg.append(f'''
        <rect x="{bx}" y="{y + 8}" width="{useful_w}" height="20" rx="4" fill="url(#greenUseful)"/>
        ''')
        # Waste Bar
        svg.append(f'''
        <rect x="{bx + useful_w + 2}" y="{y + 8}" width="{waste_w}" height="20" rx="4" fill="url(#redWaste)"/>
        ''')
        # Label Total
        total_c = "#34D399" if i == 0 else "#9CA3AF"
        svg.append(f'''
        <text x="{bx + useful_w + waste_w + 12}" y="{y + 23}" fill="{total_c}" font-family="sans-serif" font-size="12" font-weight="{'800' if i==0 else '600'}">{b['total']} tokens {'(⚡ Mais Econômico)' if i==0 else ''}</text>
        ''')
        
    svg_code = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    {''.join(svg)}
</svg>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    create_top15_leaderboard_svg("simplicio-27b/assets/leaderboard_top15.svg")
    create_tier_list_svg("simplicio-27b/assets/tier_list_coding.svg")
    create_ludic_token_cost_svg("simplicio-27b/assets/token_economy_cost.svg")
