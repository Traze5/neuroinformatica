# tools/check_encoding.py
import pathlib, sys
bad = []
for p in pathlib.Path(".").rglob("*.py"):
    try:
        p.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        bad.append((str(p), e))
for path, err in bad:
    print("NO UTF-8:", path, "-", err)
if not bad:
    print("OK: todos los .py son UTF-8")
