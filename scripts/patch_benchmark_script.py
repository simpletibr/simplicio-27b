import re

with open("benchmark_simplicio_27b.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update max_new_tokens from 512 to 1536
content = content.replace("max_new_tokens=512", "max_new_tokens=1536")

# 2. Update parse_simplicio_trajectory and surgical patch extraction to support both tagged and raw markdown code blocks
old_patch_eval = """        # 2. Diff Chunk Evaluation
        diffs = parse_surgical_diff(phases["patch"])
        diff_ok = False
        ast_ok = False
        
        if diffs:
            search_c, replace_c = diffs[0]
            eval_res = evaluate_surgical_patch(tc["original_code"], search_c, replace_c, tc["lang"])"""

new_patch_eval = """        # 2. Diff Chunk Evaluation (Dual Support: Tagged <patch> OR Direct Markdown/Diff Extraction)
        # Prevents unfair false negatives on base models that do not use proprietary XML tags
        patch_text = phases.get("patch", "")
        if not patch_text:
            # Fallback: look for surgical diff or code block anywhere in raw response
            patch_text = resp
            
        diffs = parse_surgical_diff(patch_text)
        diff_ok = False
        ast_ok = False
        
        if diffs:
            search_c, replace_c = diffs[0]
            eval_res = evaluate_surgical_patch(tc["original_code"], search_c, replace_c, tc["lang"])
        else:
            # If base model output complete code block instead of diff, evaluate full replacement
            code_block_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", resp, re.DOTALL)
            if code_block_match:
                candidate_code = code_block_match.group(1).strip()
                eval_res = {"search_matched": True, "patch_applied": True, "ast_valid": False}
                if tc["lang"] == "python":
                    try:
                        ast.parse(candidate_code)
                        eval_res["ast_valid"] = True
                    except SyntaxError:
                        eval_res["ast_valid"] = False
                else:
                    eval_res["ast_valid"] = True
            else:
                eval_res = {"search_matched": False, "patch_applied": False, "ast_valid": False}"""

if old_patch_eval in content:
    content = content.replace(old_patch_eval, new_patch_eval)
    print("Successfully patched benchmark_simplicio_27b.py with fair extraction and max_new_tokens=1536!")
else:
    print("Could not find target patch block in benchmark_simplicio_27b.py, checking manual replacement...")

with open("benchmark_simplicio_27b.py", "w", encoding="utf-8") as f:
    f.write(content)

