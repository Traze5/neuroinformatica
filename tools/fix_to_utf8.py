# tools/fix_to_utf8.py
from pathlib import Path

targets = [Path(r"pages\1_Modelo Hindmarsh–Rose.py")]

for p in targets:
    raw = p.read_bytes()
    try:
        raw.decode("utf-8")
        print(f"UTF-8 OK: {p}")
        continue
    except UnicodeDecodeError:
        pass

    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            Path(str(p) + ".bak").write_bytes(raw)  # backup
            p.write_text(text, encoding="utf-8", newline="\n")
            print(f"Convertido {p} desde {enc} → UTF-8")
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"No pude decodificar {p} con utf-8-sig/cp1252/latin-1.")
