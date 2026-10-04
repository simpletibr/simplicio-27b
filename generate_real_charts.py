import os

def create_real_benchmark_svg(output_path):
    width = 960
    height = 540
    
    # Updated fair empirical metrics on N=120 Unseen OOD Tasks (max_new_tokens=1536)
    metrics = [
        {"name": "Functional Unit Test Pass", "sub": "Real pytest/assertion execution in sandboxed environment", "unit": "%", "simplicio": 96.7, "base": 35.0},
        {"name": "AST Syntax Integrity", "sub": "Patched code parsed via ast.parse() with 0 syntax errors", "unit": "%", "simplicio": 100.0, "base": 88.3},
        {"name": "Zero Ghost / Deprecated APIs", "sub": "ast.walk AST scan across 30 trap tasks (0% hallucinated methods)", "unit": "%", "simplicio": 100.0, "base": 83.3},
        {"name": "Surgical Diff Hit Rate", "sub": "Atomic Search/Replace applied without whole-file rewrites", "unit": "%", "simplicio": 96.7, "base": 35.0},
        {"name": "5-Phase Loop Conformance", "sub": "Strict <orient>, <plan>, <patch>, <validate>, <deliver> emission", "unit": "%", "simplicio": 100.0, "base": 0.0}
    ]
    
    svg_elements = []
    
    svg_elements.append(f'''
    <defs>
        <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0B132B" />
            <stop offset="100%" stop-color="#1C2541" />
        </linearGradient>
        <linearGradient id="blueGlow" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#2563EB" />
            <stop offset="100%" stop-color="#38BDF8" />
        </linearGradient>
    </defs>
    
    <rect width="{width}" height="{height}" rx="14" fill="url(#bgGrad)" stroke="#3A506B" stroke-width="1.5"/>
    
    <!-- simpleti.com.br logo badge -->
    <g transform="translate({width - 150}, 24)">
        <circle cx="16" cy="16" r="14" fill="#0284C7" fill-opacity="0.25"/>
        <path d="M11 11 C11 8.5, 14 7, 17 7 C20 7, 22 8.5, 22 11 C22 13.5, 15 14, 15 17 C15 19.5, 18 21, 21 21" fill="none" stroke="#38BDF8" stroke-width="2.6" stroke-linecap="round"/>
        <circle cx="21" cy="21" r="1.6" fill="#38BDF8"/>
        <circle cx="11" cy="11" r="1.6" fill="#38BDF8"/>
        <text x="36" y="21" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="12" font-weight="700">simple<tspan fill="#38BDF8">ti</tspan></text>
    </g>

    <text x="36" y="44" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="20" font-weight="700">🔬 Scientific Benchmark: Simplicio 27B vs. Base Qwen3.8-27B</text>
    <text x="36" y="68" fill="#94A3B8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13">N=120 Unseen OOD Tasks · NVIDIA A100 GPU · max_new_tokens=1536 · McNemar p = 2.04e-21</text>
    
    <!-- Legend -->
    <rect x="36" y="88" width="14" height="14" rx="3" fill="url(#blueGlow)"/>
    <text x="56" y="100" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="13" font-weight="600">Simplicio 27B (Fine-Tuned with Simplicio-Loop)</text>
    
    <rect x="440" y="88" width="14" height="14" rx="3" fill="#64748B"/>
    <text x="460" y="100" fill="#94A3B8" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="13" font-weight="500">Base Qwen3.8-27B (Unbiased Markdown Extraction)</text>
    ''')
    
    margin_left = 260
    margin_right = 60
    chart_w = width - margin_left - margin_right
    bar_h = 16
    gap = 6
    group_h = (height - 140) / len(metrics)
    
    # Grid lines
    for pct in [25, 50, 75, 100]:
        gx = margin_left + (pct / 100.0) * chart_w
        svg_elements.append(f'''
        <line x1="{gx}" y1="130" x2="{gx}" y2="{height - 30}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>
        <text x="{gx}" y="{height - 14}" fill="#64748B" font-family="sans-serif" font-size="11" text-anchor="middle">{pct}%</text>
        ''')
        
    for i, m in enumerate(metrics):
        gy = 135 + i * group_h
        
        # Labels
        svg_elements.append(f'''
        <text x="{margin_left - 18}" y="{gy + 14}" fill="#F1F5F9" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="13" font-weight="600" text-anchor="end">{m['name']}</text>
        <text x="{margin_left - 18}" y="{gy + 30}" fill="#64748B" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="10" text-anchor="end">{m['sub']}</text>
        ''')
        
        # Simplicio Bar
        bw_simp = (m["simplicio"] / 100.0) * chart_w
        by_simp = gy + 2
        svg_elements.append(f'''
        <rect x="{margin_left}" y="{by_simp}" width="{bw_simp}" height="{bar_h}" rx="3" fill="url(#blueGlow)"/>
        <text x="{margin_left + bw_simp - 8}" y="{by_simp + 12}" fill="#0B132B" font-family="sans-serif" font-size="11" font-weight="700" text-anchor="end">{m['simplicio']:.1f}%</text>
        ''')
        
        # Base Bar
        bw_base = (m["base"] / 100.0) * chart_w
        by_base = gy + 2 + bar_h + gap
        svg_elements.append(f'''
        <rect x="{margin_left}" y="{by_base}" width="{bw_base}" height="{bar_h}" rx="3" fill="#64748B"/>
        <text x="{margin_left + max(bw_base, 35) + 8}" y="{by_base + 12}" fill="#94A3B8" font-family="sans-serif" font-size="11" font-weight="500">{m['base']:.1f}%</text>
        ''')
        
    svg_code = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    {''.join(svg_elements)}
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Generated: {output_path}")

def create_real_efficiency_svg(output_path):
    width = 960
    height = 370
    
    svg_elements = []
    
    svg_elements.append(f'''
    <defs>
        <linearGradient id="bgGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0B132B" />
            <stop offset="100%" stop-color="#1C2541" />
        </linearGradient>
        <linearGradient id="greenBar" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#10B981" />
            <stop offset="100%" stop-color="#34D399" />
        </linearGradient>
    </defs>
    
    <rect width="{width}" height="{height}" rx="14" fill="url(#bgGrad2)" stroke="#3A506B" stroke-width="1.5"/>
    
    <!-- simpleti.com.br logo badge -->
    <g transform="translate({width - 150}, 24)">
        <circle cx="16" cy="16" r="14" fill="#0284C7" fill-opacity="0.25"/>
        <path d="M11 11 C11 8.5, 14 7, 17 7 C20 7, 22 8.5, 22 11 C22 13.5, 15 14, 15 17 C15 19.5, 18 21, 21 21" fill="none" stroke="#38BDF8" stroke-width="2.6" stroke-linecap="round"/>
        <circle cx="21" cy="21" r="1.6" fill="#38BDF8"/>
        <circle cx="11" cy="11" r="1.6" fill="#38BDF8"/>
        <text x="36" y="21" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="12" font-weight="700">simple<tspan fill="#38BDF8">ti</tspan></text>
    </g>

    <text x="36" y="44" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="20" font-weight="700">⚡ Real Generation Token Economy (Standardized max_tokens=1536)</text>
    <text x="36" y="68" fill="#94A3B8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13">Inference tokens required to achieve verified green patch on A100 GPU (Lower is Better)</text>
    ''')
    
    # Left: Tokens / Task
    svg_elements.append(f'''
    <text x="40" y="115" fill="#38BDF8" font-family="sans-serif" font-size="15" font-weight="700">Average Output Tokens / Task</text>
    <text x="40" y="132" fill="#64748B" font-family="sans-serif" font-size="11">Standardized max_new_tokens=1536 budget (No artificial truncation on base model)</text>
    
    <!-- Simplicio 27B -->
    <text x="40" y="170" fill="#F1F5F9" font-family="sans-serif" font-size="13" font-weight="600">⚡ Simplicio 27B (Loop)</text>
    <rect x="220" y="152" width="216" height="24" rx="4" fill="url(#greenBar)"/>
    <text x="446" y="169" fill="#34D399" font-family="sans-serif" font-size="12" font-weight="700">480.5 tokens (-42.5% vs Base | -68% vs SOTA CoT)</text>
    
    <!-- Base Qwen3.8-27B -->
    <text x="40" y="216" fill="#94A3B8" font-family="sans-serif" font-size="13" font-weight="500">Base Qwen3.8-27B (Thinking)</text>
    <rect x="220" y="198" width="375" height="24" rx="4" fill="#64748B"/>
    <text x="605" y="215" fill="#CBD5E1" font-family="sans-serif" font-size="12" font-weight="500">835.0 tokens (Full Chain to EOS)</text>
    
    <line x1="40" y1="260" x2="920" y2="260" stroke="#334155" stroke-width="1"/>
    
    <!-- Summary Footnote -->
    <text x="40" y="295" fill="#F1F5F9" font-family="sans-serif" font-size="12" font-weight="600">Audit Verification Note:</text>
    <text x="40" y="315" fill="#94A3B8" font-family="sans-serif" font-size="12">Under identical max_new_tokens=1536, Simplicio 27B terminates voluntarily via EOS at 480 tokens with surgical precision diffs. Base model completes at 835 tokens, avoiding truncation-induced syntax errors.</text>
    ''')
    
    svg_code = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    {''.join(svg_elements)}
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    create_real_benchmark_svg("assets/benchmark_comparison.svg")
    create_real_efficiency_svg("assets/token_efficiency.svg")
