from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EM = chr(0x2014)
EN = chr(0x2013)
ESCAPED = [chr(92) + "u2014", chr(92) + "u2013"]


def _files(*names, suffixes=(".py", ".md", ".toml", ".txt")):
    out = []
    for name in names:
        p = ROOT / name
        if p.is_file():
            out.append(p)
        else:
            out += [f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix in suffixes
                    and "__pycache__" not in f.parts]
    return out


def test_no_literal_em_or_en_dash_in_code_and_config():
    for path in _files("src", "scripts", "tests", "pyproject.toml", "requirements.txt", ".env.example"):
        text = path.read_text(encoding="utf-8")
        assert EM not in text and EN not in text, str(path)


def test_no_escaped_dash_in_package_and_scripts():
    for path in _files("src", "scripts"):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(e in text for e in ESCAPED), str(path)
