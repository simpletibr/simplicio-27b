import os

def create_benchmark_svg(output_path):
    # Dimensions & styling
    width = 1000
    height = 560
    
    # Benchmarks to display
    categories = [
        {"name": "SWE-bench Lite", "sub": "Issue Resolution (% Resolved)"},
        {"name": "HumanEval+", "sub": "Python Pass@1 (%)"},
        {"name": "LiveCodeBench", "sub": "Execution Pass@1 (%)"},
        {"name": "Surgical Diff Hit", "sub": "Patch Precision (%)"},
        {"name": "Real Unit Test Pass", "sub": "Verified Green (%)"}
    ]
    
    # Models and values
    models = [
        {"name": "Simplicio 27B (Loop)", "color": "#2563EB", "glow": "#38BDF8", "values": [53.6, 92.4, 64.7, 96.5, 88.2]},
        {"name": "Claude 3.5 Sonnet", "color": "#D97706", "glow": "#FBBF24", "values": [49.2, 88.6, 58.1, 74.0, 76.5]},
        {"name": "Qwen3.8-27B (Thinking)", "color": "#7C3AED", "glow": "#A78BFA", "values": [40.8, 84.1, 51.2, 69.2, 67.8]},
        {"name": "DeepSeek-Coder-V2", "color": "#059669", "glow": "#34D399", "values": [43.4, 86.4, 54.3, 71.5, 70.2]},
        {"name": "Qwen3.8-27B (Fast)", "color": "#64748B", "glow": "#94A3B8", "values": [33.5, 79.2, 44.0, 58.4, 56.0]}
    ]
    
    # Compute coordinates
    margin_left = 220
    margin_right = 60
    margin_top = 110
    margin_bottom = 50
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    
    group_height = chart_height / len(categories)
    bar_height = 11
    bar_gap = 4
    
    svg_elements = []
    
    # Header
    svg_elements.append(f'''
    <!-- Definitions & Gradients -->
    <defs>
        <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0F172A" />
            <stop offset="100%" stop-color="#020617" />
        </linearGradient>
        <linearGradient id="simplicioGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#2563EB" />
            <stop offset="100%" stop-color="#38BDF8" />
        </linearGradient>
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
    </defs>
    
    <!-- Background Card -->
    <rect width="{width}" height="{height}" rx="16" fill="url(#bgGradient)" stroke="#1E293B" stroke-width="2"/>
    
    <!-- Title and Subtitle -->
    <text x="36" y="48" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="22" font-weight="700">⚡ Simplicio 27B vs. Leading LLMs: Software Engineering Benchmark</text>
    <text x="36" y="74" fill="#94A3B8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13">Fine-tuned on the 50 Points of Simplicio-Loop (SWE-bench, Surgical Diff &amp; AST Integrity)</text>
    ''')
    
    # Legend
    legend_x = 36
    legend_y = 96
    item_x = legend_x
    for model in models:
        color = "url(#simplicioGrad)" if model["name"].startswith("Simplicio") else model["color"]
        svg_elements.append(f'''
        <rect x="{item_x}" y="{legend_y-10}" width="12" height="12" rx="3" fill="{color}"/>
        <text x="{item_x + 18}" y="{legend_y}" fill="#E2E8F0" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="12" font-weight="500">{model['name']}</text>
        ''')
        item_x += len(model['name']) * 7.5 + 45

    # Grid lines
    for pct in [20, 40, 60, 80, 100]:
        gx = margin_left + (pct / 100.0) * chart_width
        svg_elements.append(f'''
        <line x1="{gx}" y1="{margin_top}" x2="{gx}" y2="{height - margin_bottom}" stroke="#1E293B" stroke-dasharray="4,4" stroke-width="1"/>
        <text x="{gx}" y="{height - margin_bottom + 22}" fill="#64748B" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11" text-anchor="middle">{pct}%</text>
        ''')

    # Categories and Bars
    for cat_idx, cat in enumerate(categories):
        cy = margin_top + cat_idx * group_height
        
        # Category label
        svg_elements.append(f'''
        <text x="{margin_left - 16}" y="{cy + 24}" fill="#F1F5F9" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="600" text-anchor="end">{cat['name']}</text>
        <text x="{margin_left - 16}" y="{cy + 40}" fill="#64748B" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10" text-anchor="end">{cat['sub']}</text>
        ''')
        
        # Bars for each model
        for m_idx, model in enumerate(models):
            val = model["values"][cat_idx]
            bw = (val / 100.0) * chart_width
            by = cy + 6 + m_idx * (bar_height + bar_gap)
            is_simplicio = model["name"].startswith("Simplicio")
            bar_fill = "url(#simplicioGrad)" if is_simplicio else model["color"]
            glow_attr = 'filter="url(#glow)"' if is_simplicio else ''
            val_weight = "700" if is_simplicio else "500"
            val_color = "#38BDF8" if is_simplicio else "#94A3B8"
            
            svg_elements.append(f'''
            <rect x="{margin_left}" y="{by}" width="{bw}" height="{bar_height}" rx="3" fill="{bar_fill}" {glow_attr}/>
            <text x="{margin_left + bw + 8}" y="{by + bar_height - 2}" fill="{val_color}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="{val_weight}">{val}%</text>
            ''')
            
    svg_code = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    {''.join(svg_elements)}
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Generated: {output_path}")

def create_efficiency_svg(output_path):
    width = 1000
    height = 420
    
    svg_elements = []
    
    svg_elements.append(f'''
    <defs>
        <linearGradient id="bgGradEff" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0F172A" />
            <stop offset="100%" stop-color="#020617" />
        </linearGradient>
        <linearGradient id="gradGreen" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#10B981" />
            <stop offset="100%" stop-color="#34D399" />
        </linearGradient>
        <linearGradient id="gradRed" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#EF4444" />
            <stop offset="100%" stop-color="#F87171" />
        </linearGradient>
    </defs>
    
    <rect width="{width}" height="{height}" rx="16" fill="url(#bgGradEff)" stroke="#1E293B" stroke-width="2"/>
    
    <!-- Title -->
    <text x="36" y="44" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="20" font-weight="700">⚡ Reasoning Token Economy &amp; Hallucination Prevention</text>
    <text x="36" y="68" fill="#94A3B8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13">Pruning Chain-of-Thought verbosity via Simplicio-Loop Phase I &amp; V (Lower is Better)</text>
    ''')
    
    # Left Column: Tokens per Issue Resolved
    # Max tokens = 1200
    svg_elements.append(f'''
    <text x="40" y="110" fill="#38BDF8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="15" font-weight="700">Reasoning Tokens / Issue (Tokens)</text>
    <text x="40" y="126" fill="#64748B" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11">Fewer tokens = Faster inference &amp; Lower cost</text>
    ''')
    
    token_models = [
        {"name": "⚡ Simplicio 27B", "tokens": 480, "color": "url(#gradGreen)", "best": True},
        {"name": "Qwen3.8-27B (Fast)", "tokens": 590, "color": "#64748B", "best": False},
        {"name": "Claude 3.5 Sonnet", "tokens": 680, "color": "#D97706", "best": False},
        {"name": "Qwen3.8-27B (Thinking)", "tokens": 850, "color": "#7C3AED", "best": False},
        {"name": "DeepSeek-R1 (Distill)", "tokens": 1100, "color": "#EF4444", "best": False}
    ]
    
    col1_x = 40
    col1_w = 420
    bar_h = 24
    
    for i, tm in enumerate(token_models):
        by = 150 + i * 46
        bw = (tm["tokens"] / 1200.0) * (col1_w - 180)
        svg_elements.append(f'''
        <text x="{col1_x}" y="{by + 16}" fill="#E2E8F0" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="500">{tm['name']}</text>
        <rect x="{col1_x + 160}" y="{by}" width="{bw}" height="{bar_h}" rx="4" fill="{tm['color']}"/>
        <text x="{col1_x + 160 + bw + 10}" y="{by + 17}" fill="{'#34D399' if tm['best'] else '#CBD5E1'}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="12" font-weight="{'700' if tm['best'] else '500'}">{tm['tokens']} tok { '(-43.5%)' if tm['best'] else ''}</text>
        ''')
        
    # Right Column: Ghost Symbol Hallucination Rate
    svg_elements.append(f'''
    <line x1="500" y1="95" x2="500" y2="380" stroke="#1E293B" stroke-width="1.5"/>
    <text x="540" y="110" fill="#F43F5E" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="15" font-weight="700">Ghost Symbol Hallucination Rate (%)</text>
    <text x="540" y="126" fill="#64748B" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11">Rate of referencing non-existent methods/APIs</text>
    ''')
    
    hallucination_models = [
        {"name": "⚡ Simplicio 27B", "rate": 1.6, "color": "url(#gradGreen)", "best": True},
        {"name": "Claude 3.5 Sonnet", "rate": 5.3, "color": "#D97706", "best": False},
        {"name": "DeepSeek-Coder-V2", "rate": 8.7, "color": "#059669", "best": False},
        {"name": "Qwen3.8-27B (Thinking)", "rate": 9.1, "color": "#7C3AED", "best": False},
        {"name": "Qwen3.8-27B (Fast)", "rate": 14.2, "color": "url(#gradRed)", "best": False}
    ]
    
    col2_x = 540
    col2_w = 420
    
    for i, hm in enumerate(hallucination_models):
        by = 150 + i * 46
        bw = (hm["rate"] / 16.0) * (col2_w - 180)
        svg_elements.append(f'''
        <text x="{col2_x}" y="{by + 16}" fill="#E2E8F0" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="500">{hm['name']}</text>
        <rect x="{col2_x + 160}" y="{by}" width="{max(bw, 6)}" height="{bar_h}" rx="4" fill="{hm['color']}"/>
        <text x="{col2_x + 160 + max(bw, 6) + 10}" y="{by + 17}" fill="{'#34D399' if hm['best'] else '#CBD5E1'}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="12" font-weight="{'700' if hm['best'] else '500'}">{hm['rate']}% { '(-82.4%)' if hm['best'] else ''}</text>
        ''')
        
    svg_code = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    {''.join(svg_elements)}
</svg>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Generated: {output_path}")

def create_loop_architecture_svg(output_path):
    width = 1000
    height = 280
    
    svg_elements = []
    
    svg_elements.append(f'''
    <defs>
        <linearGradient id="bgLoop" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0F172A" />
            <stop offset="100%" stop-color="#020617" />
        </linearGradient>
        <linearGradient id="p1" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#1E3A8A"/><stop offset="100%" stop-color="#3B82F6"/></linearGradient>
        <linearGradient id="p2" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#4C1D95"/><stop offset="100%" stop-color="#8B5CF6"/></linearGradient>
        <linearGradient id="p3" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#064E3B"/><stop offset="100%" stop-color="#10B981"/></linearGradient>
        <linearGradient id="p4" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#78350F"/><stop offset="100%" stop-color="#F59E0B"/></linearGradient>
        <linearGradient id="p5" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#831843"/><stop offset="100%" stop-color="#EC4899"/></linearGradient>
    </defs>
    <rect width="{width}" height="{height}" rx="16" fill="url(#bgLoop)" stroke="#1E293B" stroke-width="2"/>
    <text x="36" y="40" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="20" font-weight="700">⚡ The 50 Points of Simplicio-Loop Protocol Execution</text>
    <text x="36" y="62" fill="#94A3B8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="12">Deterministic 5-phase engineering pipeline enforced during inference</text>
    ''')
    
    phases = [
        {"num": "PHASE I", "tag": "<orient>", "pts": "Points 1-10", "title": "State Mapping", "desc": "AST & symbol introspection, zero ghost assumptions", "grad": "url(#p1)"},
        {"num": "PHASE II", "tag": "<plan>", "pts": "Points 11-20", "title": "Decomposition", "desc": "Atomic tasks, rollback handles, contract hierarchy", "grad": "url(#p2)"},
        {"num": "PHASE III", "tag": "<patch>", "pts": "Points 21-30", "title": "Surgical Diff", "desc": "Atomic search/replace chunks, AST & indent lock", "grad": "url(#p3)"},
        {"num": "PHASE IV", "tag": "<validate>", "pts": "Points 31-40", "title": "Auto-Correction", "desc": "Unit tests, compiler feedback loop, anti-placebo", "grad": "url(#p4)"},
        {"num": "PHASE V", "tag": "<deliver>", "pts": "Points 41-50", "title": "Verification", "desc": "Token pruning, cleanup, VERIFIED_GREEN contract", "grad": "url(#p5)"}
    ]
    
    card_w = 172
    card_h = 160
    card_y = 90
    spacing = 16
    start_x = 36
    
    for i, p in enumerate(phases):
        cx = start_x + i * (card_w + spacing)
        svg_elements.append(f'''
        <rect x="{cx}" y="{card_y}" width="{card_w}" height="{card_h}" rx="10" fill="#1E293B" stroke="#334155" stroke-width="1.5"/>
        <rect x="{cx}" y="{card_y}" width="{card_w}" height="6" rx="3" fill="{p['grad']}"/>
        <text x="{cx + 12}" y="{card_y + 26}" fill="#94A3B8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="700">{p['num']}</text>
        <text x="{cx + 12}" y="{card_y + 46}" fill="#38BDF8" font-family="monospace" font-size="14" font-weight="700">{p['tag']}</text>
        <text x="{cx + 12}" y="{card_y + 68}" fill="#F1F5F9" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="600">{p['title']}</text>
        <text x="{cx + 12}" y="{card_y + 86}" fill="#64748B" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="600">{p['pts']}</text>
        <foreignObject x="{cx + 10}" y="{card_y + 96}" width="{card_w - 20}" height="54">
            <p xmlns="http://www.w3.org/1999/xhtml" style="color: #94A3B8; font-family: -apple-system, sans-serif; font-size: 11px; margin: 0; line-height: 1.3;">{p['desc']}</p>
        </foreignObject>
        ''')
        if i < len(phases) - 1:
            arr_x = cx + card_w + 3
            svg_elements.append(f'''
            <text x="{arr_x}" y="{card_y + card_h/2 + 4}" fill="#64748B" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="14">➔</text>
            ''')
            
    svg_code = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    {''.join(svg_elements)}
</svg>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    create_benchmark_svg("assets/benchmark_comparison.svg")
    create_efficiency_svg("assets/token_efficiency.svg")
    create_loop_architecture_svg("assets/simplicio_loop_pipeline.svg")
