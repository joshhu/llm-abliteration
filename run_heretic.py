import os, sys
from questionary import Choice
import heretic.main as M

SAVE_DIR = os.environ.get("HERETIC_SAVE_DIR", "/heretic/out/Qwen3.8-27B-abliterated")
seen = {"trial": 0, "action": 0, "saved": False}

def _vals(choices):
    return [c.value if isinstance(c, Choice) else c for c in choices]

def prompt_select(message, choices, **kw):
    vals = _vals(choices)
    print("\n[auto-prompt] %r (%d options)" % (message, len(vals)), flush=True)
    for i, v in enumerate(vals, 1):
        print("[auto-prompt]   [%d] %s" % (i, getattr(v, "title", v)), flush=True)
    m = message.lower()
    if "which trial" in m:
        seen["trial"] += 1
        if seen["saved"] or seen["trial"] > 1:
            print("[auto-prompt] model already saved -> exiting", flush=True)
            raise SystemExit(0)
        print("[auto-prompt] -> option 1 (best Pareto trial)", flush=True)
        return vals[0]
    if "what do you want to do" in m:
        seen["action"] += 1
        if seen["action"] == 1:
            for v in vals:
                if isinstance(v, str) and v.startswith("Save"):
                    print("[auto-prompt] -> Save to local folder", flush=True)
                    return v
        print("[auto-prompt] -> back to menu", flush=True)
        return ""
    print("[auto-prompt] -> option 1 (default)", flush=True)
    return vals[0]

def prompt_path(message):
    print("\n[auto-prompt] %r -> %s" % (message, SAVE_DIR), flush=True)
    seen["saved"] = True
    return SAVE_DIR

def prompt_text(message, default="", **kw):
    print("\n[auto-prompt] %r -> %r" % (message, default), flush=True)
    return default

M.prompt_select = prompt_select
M.prompt_path = prompt_path
M.prompt_text = prompt_text

if __name__ == "__main__":
    sys.exit(M.main())
