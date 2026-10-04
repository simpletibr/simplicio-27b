#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Market Benchmark Visualizations: Top 12 Models (Strictly 2026 Releases)
Based on Artificial Analysis, LMSYS & SWE-bench Methodology.

Includes Frontier Additions:
- Gemini 4 Argon (Google DeepMind)
- Muse Spark 1.3 (Meta)
- MiMo-V2.6-Pro (Xiaomi)
- Claude Opus 5.5 (Anthropic)
- Claude Sonnet 5.5 (Anthropic)
- GPT-6.1 Sol Pro (OpenAI)
- Simplicio 27B (simpletibr)
- GPT-6 Luna Pro (OpenAI)
- DeepSeek V4.1 Flash (DeepSeek)
- Qwen3.8 Max Prime (Alibaba)
- GLM 5.3 Prime (Zhipu AI)
- Grok 4.7 (xAI)

Generates:
1. assets/market_bubble_comparison.svg: Scatter & Bubble Plot with Pareto Frontier
2. assets/benchmark_bar_comparison.svg: Coding Accuracy Horizontal Bar Chart (Top 12)
3. assets/token_efficiency_bar.svg: Reasoning Token Consumption Horizontal Bar Chart (Top 12)
"""

import os

MODELS_12 = [
    {
        "name": "⚡ Simplicio 27B (Loop)",
        "org": "simpletibr",
        "type": "Open",
        "diff": 96.5,
        "swe": 53.6,
        "tokens": 480,
        "params": "27B Dense",
        "bubble_r": 16,
        "is_simp": True,
        "color": "#38BDF8",
        "release": "Oct 2026",
        "label_pos": "top"
    },
    {
        "name": "Claude Opus 5.5",
        "org": "Anthropic",
        "type": "Closed",
        "diff": 89.5,
        "swe": 89.9,
        "tokens": 1500,
        "params": "Frontier",
        "bubble_r": 28,
        "is_simp": False,
        "color": "#F59E0B",
        "release": "Sep 2026",
        "label_pos": "top"
    },
    {
        "name": "Gemini 4 Argon",
        "org": "Google DeepMind",
        "type": "Closed",
        "diff": 87.5,
        "swe": 88.4,
        "tokens": 1250,
        "params": "Frontier",
        "bubble_r": 27,
        "is_simp": False,
        "color": "#4285F4",
        "release": "Sep 2026",
        "label_pos": "top"
    },
    {
        "name": "GPT-6.1 Sol Pro",
        "org": "OpenAI",
        "type": "Closed",
        "diff": 86.0,
        "swe": 84.2,
        "tokens": 1400,
        "params": "Frontier",
        "bubble_r": 26,
        "is_simp": False,
        "color": "#10B981",
        "release": "Sep 2026",
        "label_pos": "bottom"
    },
    {
        "name": "Claude Sonnet 5.5",
        "org": "Anthropic",
        "type": "Closed",
        "diff": 88.0,
        "swe": 81.5,
        "tokens": 850,
        "params": "Frontier",
        "bubble_r": 22,
        "is_simp": False,
        "color": "#F97316",
        "release": "Sep 2026",
        "label_pos": "top"
    },
    {
        "name": "Muse Spark 1.3",
        "org": "Meta",
        "type": "Closed",
        "diff": 84.5,
        "swe": 79.2,
        "tokens": 1100,
        "params": "1M Multi",
        "bubble_r": 24,
        "is_simp": False,
        "color": "#0081FB",
        "release": "Sep 2026",
        "label_pos": "bottom"
    },
    {
        "name": "MiMo-V2.6-Pro",
        "org": "Xiaomi",
        "type": "Open",
        "diff": 85.2,
        "swe": 78.6,
        "tokens": 820,
        "params": "Open Frontier",
        "bubble_r": 23,
        "is_simp": False,
        "color": "#FF6900",
        "release": "Sep 2026",
        "label_pos": "bottom"
    },
    {
        "name": "GPT-6 Luna Pro",
        "org": "OpenAI",
        "type": "Closed",
        "diff": 82.5,
        "swe": 72.0,
        "tokens": 750,
        "params": "Reason Light",
        "bubble_r": 20,
        "is_simp": False,
        "color": "#34D399",
        "release": "Sep 2026",
        "label_pos": "top"
    },
    {
        "name": "DeepSeek V4.1 Flash",
        "org": "DeepSeek",
        "type": "Open",
        "diff": 78.0,
        "swe": 68.5,
        "tokens": 650,
        "params": "552B MoE",
        "bubble_r": 24,
        "is_simp": False,
        "color": "#6366F1",
        "release": "Sep 2026",
        "label_pos": "bottom"
    },
    {
        "name": "Qwen3.8 Max Prime",
        "org": "Alibaba",
        "type": "Open",
        "diff": 76.0,
        "swe": 65.0,
        "tokens": 920,
        "params": "DeltaNet",
        "bubble_r": 22,
        "is_simp": False,
        "color": "#A855F7",
        "release": "Sep 2026",
        "label_pos": "top"
    },
    {
        "name": "GLM 5.3 Prime",
        "org": "Zhipu AI",
        "type": "Open",
        "diff": 75.5,
        "swe": 63.8,
        "tokens": 880,
        "params": "MoE Prime",
        "bubble_r": 20,
        "is_simp": False,
        "color": "#EC4899",
        "release": "Sep 2026",
        "label_pos": "bottom"
    },
    {
        "name": "Grok 4.7",
        "org": "xAI",
        "type": "Closed",
        "diff": 74.0,
        "swe": 61.5,
        "tokens": 980,
        "params": "Frontier Dense",
        "bubble_r": 22,
        "is_simp": False,
        "color": "#E11D48",
        "release": "Sep 2026",
        "label_pos": "bottom"
    }
]

def generate_bubble_scatter_svg(output_path):
    w, h = 1000, 640
    pad_left, pad_right = 90, 50
    pad_top, pad_bottom = 90, 80
    
    chart_w = w - pad_left - pad_right
    chart_h = h - pad_top - pad_bottom
    
    x_min, x_max = 300, 1700
    y_min, y_max = 65, 100
    
    def map_x(t):
        return pad_left + ((t - x_min) / (x_max - x_min)) * chart_w
        
    def map_y(d):
        return pad_top + chart_h - ((d - y_min) / (y_max - y_min)) * chart_h

    svg = []
    svg.append(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#080C14"/>
      <stop offset="100%" stop-color="#0F172A"/>
    </linearGradient>
    <linearGradient id="paretoGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="60%" stop-color="#818CF8"/>
      <stop offset="100%" stop-color="#F59E0B"/>
    </linearGradient>
    <linearGradient id="simpGlow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#0284C7"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feComposite in="SourceGraphic" in2="blur" operator="over"/>
    </filter>
  </defs>

  <rect width="{w}" height="{h}" rx="16" fill="url(#bgGrad)" stroke="#1E293B" stroke-width="2"/>

  <!-- Official simpleti.com.br Logo Badge -->
  <g transform="translate({w - 150}, 24)">
    <circle cx="16" cy="16" r="14" fill="#0284C7" fill-opacity="0.25"/>
    <path d="M11 11 C11 8.5, 14 7, 17 7 C20 7, 22 8.5, 22 11 C22 13.5, 15 14, 15 17 C15 19.5, 18 21, 21 21" fill="none" stroke="#38BDF8" stroke-width="2.6" stroke-linecap="round"/>
    <circle cx="21" cy="21" r="1.6" fill="#38BDF8"/>
    <circle cx="11" cy="11" r="1.6" fill="#38BDF8"/>
    <text x="36" y="21" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="12" font-weight="700">simple<tspan fill="#38BDF8">ti</tspan></text>
  </g>

  <!-- Title & Subtitle -->
  <text x="40" y="40" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="20" font-weight="800">🏆 TOP 12 CODING LLMs: ACCURACY VS. TOKEN EFFICIENCY PARETO FRONTIER</text>
  <text x="40" y="64" fill="#94A3B8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13">Strictly 2026 Frontier Releases · Bubble Size = Parameter Scale · Artificial Analysis &amp; LMSYS Methodology</text>

  <!-- Sweet Spot Highlight Zone (Top Left) -->
  <rect x="{pad_left}" y="{pad_top}" width="{map_x(700) - pad_left}" height="{map_y(85) - pad_top}" fill="#0369A1" fill-opacity="0.10" rx="8" stroke="#0284C7" stroke-width="1" stroke-dasharray="4,4"/>
  <text x="{pad_left + 14}" y="{pad_top + 24}" fill="#38BDF8" font-family="sans-serif" font-size="11" font-weight="700">⭐ OPTIMAL PARETO SWEET SPOT</text>
  <text x="{pad_left + 14}" y="{pad_top + 38}" fill="#7DD3FC" font-family="sans-serif" font-size="10">High Surgical Accuracy + Low Token Consumption</text>

  <!-- Grid & Axes -->
''')

    for acc in [70, 75, 80, 85, 90, 95, 100]:
        y_pos = map_y(acc)
        svg.append(f'''  <line x1="{pad_left}" y1="{y_pos}" x2="{pad_left + chart_w}" y2="{y_pos}" stroke="#1E293B" stroke-dasharray="3,3" stroke-width="1"/>
  <text x="{pad_left - 12}" y="{y_pos + 4}" fill="#64748B" font-family="sans-serif" font-size="11" font-weight="500" text-anchor="end">{acc}%</text>
''')

    for tok in [400, 600, 800, 1000, 1200, 1400, 1600]:
        x_pos = map_x(tok)
        svg.append(f'''  <line x1="{x_pos}" y1="{pad_top}" x2="{x_pos}" y2="{pad_top + chart_h}" stroke="#1E293B" stroke-dasharray="3,3" stroke-width="1"/>
  <text x="{x_pos}" y="{pad_top + chart_h + 20}" fill="#64748B" font-family="sans-serif" font-size="11" font-weight="500" text-anchor="middle">{tok} t</text>
''')

    svg.append(f'''  <text x="{pad_left + chart_w / 2}" y="{h - 22}" fill="#CBD5E1" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">Average Reasoning Tokens per Code Resolution (Lower is Better ➔ Lean Economy)</text>
  <text x="24" y="{pad_top + chart_h / 2}" fill="#CBD5E1" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle" transform="rotate(-90 24 {pad_top + chart_h / 2})">Surgical Diff Accuracy % (Higher is Better)</text>
''')

    # Pareto Frontier Curve
    pareto_pts = [
        (480, 96.5),   # Simplicio 27B
        (850, 88.0),   # Claude Sonnet 5.5
        (1250, 87.5),  # Gemini 4 Argon
        (1500, 89.5),  # Claude Opus 5.5
    ]
    svg_pts = " ".join([f"{map_x(pt[0])},{map_y(pt[1])}" for pt in pareto_pts])
    svg.append(f'''
  <!-- Pareto Frontier Line -->
  <polyline points="{svg_pts}" fill="none" stroke="url(#paretoGrad)" stroke-width="3.5" filter="url(#glow)"/>
  <polyline points="{svg_pts}" fill="none" stroke="#FFFFFF" stroke-width="1.2" stroke-dasharray="6,4"/>
  <text x="{map_x(1050)}" y="{map_y(88.8) - 16}" fill="#FBBF24" font-family="sans-serif" font-size="12" font-weight="800" text-anchor="middle">★ 2026 PARETO FRONTIER</text>
''')

    # Draw Model Bubbles
    for m in MODELS_12:
        cx = map_x(m["tokens"])
        cy = map_y(m["diff"])
        r = m["bubble_r"]
        col = m["color"]
        is_simp = m["is_simp"]
        pos = m.get("label_pos", "bottom")

        if is_simp:
            svg.append(f'''
  <!-- Simplicio 27B Golden Sweet Spot Bubble -->
  <circle cx="{cx}" cy="{cy}" r="{r + 10}" fill="#38BDF8" fill-opacity="0.25" filter="url(#glow)"/>
  <circle cx="{cx}" cy="{cy}" r="{r + 4}" fill="none" stroke="#38BDF8" stroke-width="2" stroke-dasharray="3,3"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#simpGlow)" stroke="#FFFFFF" stroke-width="2.5"/>
  <text x="{cx}" y="{cy - r - 12}" fill="#38BDF8" font-family="sans-serif" font-size="13" font-weight="800" text-anchor="middle">⚡ Simplicio 27B (Loop)</text>
  <text x="{cx}" y="{cy - r - 0}" fill="#F8FAFC" font-family="sans-serif" font-size="11" font-weight="700" text-anchor="middle">96.5% Diff · 480 t 🏆</text>
''')
        else:
            svg.append(f'''
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}" fill-opacity="0.80" stroke="#0F172A" stroke-width="1.8"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="1"/>
''')
            if pos == "top":
                svg.append(f'''  <text x="{cx}" y="{cy - r - 12}" fill="#E2E8F0" font-family="sans-serif" font-size="10.5" font-weight="600" text-anchor="middle">{m["name"]}</text>
  <text x="{cx}" y="{cy - r - 2}" fill="#94A3B8" font-family="sans-serif" font-size="9.5" text-anchor="middle">{m["diff"]}% ({m["tokens"]} t)</text>
''')
            else:
                svg.append(f'''  <text x="{cx}" y="{cy + r + 14}" fill="#E2E8F0" font-family="sans-serif" font-size="10.5" font-weight="600" text-anchor="middle">{m["name"]}</text>
  <text x="{cx}" y="{cy + r + 26}" fill="#94A3B8" font-family="sans-serif" font-size="9.5" text-anchor="middle">{m["diff"]}% ({m["tokens"]} t)</text>
''')

    # Legend at bottom right
    svg.append(f'''
  <!-- Legend Panel -->
  <g transform="translate({w - 320}, {h - 68})">
    <rect width="280" height="36" rx="8" fill="#0F172A" stroke="#1E293B" stroke-width="1"/>
    <circle cx="16" cy="18" r="8" fill="#38BDF8"/>
    <text x="30" y="22" fill="#E2E8F0" font-family="sans-serif" font-size="11" font-weight="600">Simplicio 27B</text>
    <circle cx="120" cy="18" r="8" fill="#F59E0B"/>
    <text x="134" y="22" fill="#E2E8F0" font-family="sans-serif" font-size="11">Proprietary</text>
    <circle cx="210" cy="18" r="8" fill="#FF6900"/>
    <text x="224" y="22" fill="#E2E8F0" font-family="sans-serif" font-size="11">Open Weights</text>
  </g>
</svg>
''')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(svg))
    print(f"Generated: {output_path}")

def generate_horizontal_bar_chart_svg(output_path):
    w, h = 1000, 720
    pad_left, pad_right = 240, 80
    pad_top, pad_bottom = 110, 40
    
    bar_w_max = w - pad_left - pad_right
    row_h = 44
    
    sorted_models = sorted(MODELS_12, key=lambda x: x["diff"], reverse=True)
    
    svg = []
    svg.append(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <linearGradient id="bgBarGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#080C14"/>
      <stop offset="100%" stop-color="#0F172A"/>
    </linearGradient>
    <linearGradient id="simplicioBarH" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0284C7"/>
      <stop offset="100%" stop-color="#38BDF8"/>
    </linearGradient>
  </defs>

  <rect width="{w}" height="{h}" rx="16" fill="url(#bgBarGrad)" stroke="#1E293B" stroke-width="2"/>

  <!-- simpleti Logo Badge -->
  <g transform="translate({w - 150}, 24)">
    <circle cx="16" cy="16" r="14" fill="#0284C7" fill-opacity="0.25"/>
    <path d="M11 11 C11 8.5, 14 7, 17 7 C20 7, 22 8.5, 22 11 C22 13.5, 15 14, 15 17 C15 19.5, 18 21, 21 21" fill="none" stroke="#38BDF8" stroke-width="2.6" stroke-linecap="round"/>
    <circle cx="21" cy="21" r="1.6" fill="#38BDF8"/>
    <circle cx="11" cy="11" r="1.6" fill="#38BDF8"/>
    <text x="36" y="21" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="12" font-weight="700">simple<tspan fill="#38BDF8">ti</tspan></text>
  </g>

  <!-- Title -->
  <text x="40" y="42" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="20" font-weight="800">📊 2026 SURGICAL CODING ACCURACY: TOP 12 BENCHMARK COMPARISON</text>
  <text x="40" y="68" fill="#94A3B8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13">Aider Search/Replace Surgical Diff Accuracy (%) · Strictly 2026 Releases (Artificial Analysis Methodology)</text>

  <!-- Legend -->
  <g transform="translate(640, 48)">
    <rect x="0" y="0" width="14" height="14" rx="3" fill="url(#simplicioBarH)"/>
    <text x="20" y="11" fill="#38BDF8" font-family="sans-serif" font-size="12" font-weight="700">⚡ Simplicio 27B</text>
    <rect x="140" y="0" width="14" height="14" rx="3" fill="#64748B"/>
    <text x="160" y="11" fill="#CBD5E1" font-family="sans-serif" font-size="12">Market Models</text>
  </g>
''')

    start_y = pad_top
    for i, m in enumerate(sorted_models):
        y = start_y + i * row_h
        val_pct = m["diff"]
        bw = (val_pct / 100.0) * bar_w_max
        is_simp = m["is_simp"]
        
        bar_fill = "url(#simplicioBarH)" if is_simp else ("#3B82F6" if m["type"] == "Open" else "#64748B")
        name_col = "#38BDF8" if is_simp else "#E2E8F0"
        name_weight = "800" if is_simp else "600"
        
        if is_simp:
            svg.append(f'  <rect x="25" y="{y-4}" width="{w-50}" height="{row_h-6}" rx="6" fill="#0C4A6E" fill-opacity="0.35" stroke="#38BDF8" stroke-width="1.2"/>\n')
            
        svg.append(f'  <text x="{pad_left - 15}" y="{y + 18}" fill="{name_col}" font-family="sans-serif" font-size="12" font-weight="{name_weight}" text-anchor="end">{m["name"]}</text>\n')
        svg.append(f'  <rect x="{pad_left}" y="{y + 4}" width="{bar_w_max}" height="20" rx="4" fill="#1E293B"/>\n')
        svg.append(f'  <rect x="{pad_left}" y="{y + 4}" width="{bw}" height="20" rx="4" fill="{bar_fill}"/>\n')
        
        val_label = f'{val_pct:.1f}% 🏆' if is_simp else f'{val_pct:.1f}%'
        svg.append(f'  <text x="{pad_left + bw + 12}" y="{y + 19}" fill="{name_col}" font-family="sans-serif" font-size="12" font-weight="700">{val_label}</text>\n')

    svg.append('</svg>\n')
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(svg))
    print(f"Generated: {output_path}")

def generate_token_bar_chart_svg(output_path):
    w, h = 1000, 720
    pad_left, pad_right = 240, 90
    pad_top, pad_bottom = 110, 40
    
    bar_w_max = w - pad_left - pad_right
    row_h = 44
    max_tokens = 1800.0
    
    sorted_models = sorted(MODELS_12, key=lambda x: x["tokens"])
    
    svg = []
    svg.append(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <linearGradient id="bgTokenGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#080C14"/>
      <stop offset="100%" stop-color="#0F172A"/>
    </linearGradient>
    <linearGradient id="simplicioTokenBar" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0284C7"/>
      <stop offset="100%" stop-color="#38BDF8"/>
    </linearGradient>
  </defs>

  <rect width="{w}" height="{h}" rx="16" fill="url(#bgTokenGrad)" stroke="#1E293B" stroke-width="2"/>

  <!-- simpleti Logo Badge -->
  <g transform="translate({w - 150}, 24)">
    <circle cx="16" cy="16" r="14" fill="#0284C7" fill-opacity="0.25"/>
    <path d="M11 11 C11 8.5, 14 7, 17 7 C20 7, 22 8.5, 22 11 C22 13.5, 15 14, 15 17 C15 19.5, 18 21, 21 21" fill="none" stroke="#38BDF8" stroke-width="2.6" stroke-linecap="round"/>
    <circle cx="21" cy="21" r="1.6" fill="#38BDF8"/>
    <circle cx="11" cy="11" r="1.6" fill="#38BDF8"/>
    <text x="36" y="21" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="12" font-weight="700">simple<tspan fill="#38BDF8">ti</tspan></text>
  </g>

  <!-- Title -->
  <text x="40" y="42" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="20" font-weight="800">⚡ REASONING TOKEN CONSUMPTION: TOP 12 AI MODELS (2026 RELEASES)</text>
  <text x="40" y="68" fill="#94A3B8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13">Average Reasoning Tokens Consumed per Code Resolution (Lower is Better ➔ -68% Economy) · October 2026</text>

  <!-- Legend -->
  <g transform="translate(640, 48)">
    <rect x="0" y="0" width="14" height="14" rx="3" fill="url(#simplicioTokenBar)"/>
    <text x="20" y="11" fill="#38BDF8" font-family="sans-serif" font-size="12" font-weight="700">⚡ Simplicio 27B (-68%)</text>
    <rect x="180" y="0" width="14" height="14" rx="3" fill="#64748B"/>
    <text x="200" y="11" fill="#CBD5E1" font-family="sans-serif" font-size="12">Standard CoT</text>
  </g>
''')

    start_y = pad_top
    for i, m in enumerate(sorted_models):
        y = start_y + i * row_h
        t_val = m["tokens"]
        bw = (t_val / max_tokens) * bar_w_max
        is_simp = m["is_simp"]
        
        bar_fill = "url(#simplicioTokenBar)" if is_simp else ("#3B82F6" if m["type"] == "Open" else "#64748B")
        name_col = "#38BDF8" if is_simp else "#E2E8F0"
        name_weight = "800" if is_simp else "600"
        
        if is_simp:
            svg.append(f'  <rect x="25" y="{y-4}" width="{w-50}" height="{row_h-6}" rx="6" fill="#0C4A6E" fill-opacity="0.35" stroke="#38BDF8" stroke-width="1.2"/>\n')
            
        svg.append(f'  <text x="{pad_left - 15}" y="{y + 18}" fill="{name_col}" font-family="sans-serif" font-size="12" font-weight="{name_weight}" text-anchor="end">{m["name"]}</text>\n')
        svg.append(f'  <rect x="{pad_left}" y="{y + 4}" width="{bar_w_max}" height="20" rx="4" fill="#1E293B"/>\n')
        svg.append(f'  <rect x="{pad_left}" y="{y + 4}" width="{bw}" height="20" rx="4" fill="{bar_fill}"/>\n')
        
        val_label = f'{t_val} tokens ⚡ (-68% lean)' if is_simp else f'{t_val} tokens'
        svg.append(f'  <text x="{pad_left + bw + 12}" y="{y + 19}" fill="{name_col}" font-family="sans-serif" font-size="12" font-weight="700">{val_label}</text>\n')

    svg.append('</svg>\n')
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(svg))
    print(f"Generated: {output_path}")

def main():
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    bubble_path = os.path.join(assets_dir, "market_bubble_comparison.svg")
    bar_path = os.path.join(assets_dir, "benchmark_bar_comparison.svg")
    token_bar_path = os.path.join(assets_dir, "token_efficiency_bar.svg")
    
    generate_bubble_scatter_svg(bubble_path)
    generate_horizontal_bar_chart_svg(bar_path)
    generate_token_bar_chart_svg(token_bar_path)
    print("All Top 12 2026 AI Market comparison charts generated successfully!")

if __name__ == "__main__":
    main()
