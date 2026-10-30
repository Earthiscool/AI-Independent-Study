"""Rebuild the NHSJS PDF from NHSJS_Manuscript_Standard.md: python3 build_pdf.py (run from Q1/manuscript)."""
import html
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
src, dst = HERE / "NHSJS_Manuscript_Standard.md", HERE / "NHSJS_Manuscript_Standard.html"
text = re.sub(r"<!--.*?-->", "", src.read_text(), flags=re.S).strip()


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"&lt;(/?)sup&gt;", r"<\1sup>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def table(block):
    rows = [r.strip().strip("|").split("|") for r in block.strip().split("\n")]
    head, body = rows[0], [r for r in rows[2:]]
    out = ["<table>", "<thead><tr>" + "".join(f"<th>{inline(c.strip())}</th>" for c in head) + "</tr></thead>", "<tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c.strip())}</td>" for c in r) + "</tr>")
    out += ["</tbody>", "</table>"]
    return "\n".join(out)


parts = []
pending_caption = None
in_references = False
for block in re.split(r"\n\s*\n", text):
    block = block.strip()
    if not block:
        continue
    if block.startswith("#"):
        level = len(block) - len(block.lstrip("#"))
        title = block.lstrip("#").strip()
        in_references = title == "References"
        tag = "h1" if level == 1 else f"h{min(level, 4)}"
        parts.append(f"<{tag}>{inline(title)}</{tag}>")
    elif block.startswith("AUTHORS:"):
        parts.append(f'<p class="authors">{inline(block[8:])}</p>')
    elif block.startswith("AFFIL:"):
        parts.append(f'<p class="affil">{inline(block[6:])}</p>')
    elif block.startswith("CAPTION:"):
        pending_caption = inline(block[8:])
        # A caption directly after a figure belongs to that figure (caption below the figure).
        if parts and parts[-1].startswith("<figure"):
            parts[-1] = parts[-1].replace("</figure>", f"<figcaption>{pending_caption}</figcaption></figure>")
            pending_caption = None
    elif block.startswith("!["):
        m = re.match(r"!\[(.*?)\]\((.*?)\)(?:\{width=(.*?)\})?", block)
        alt, path, width = m.group(1), m.group(2), m.group(3) or "100%"
        parts.append(f'<figure><img src="{path}" alt="{html.escape(alt)}" style="width:{width}"></figure>')
    elif block.startswith("|"):
        caption = f'<div class="tcaption">{pending_caption}</div>' if pending_caption else ""
        pending_caption = None
        parts.append(f'<div class="table-wrap">{caption}{table(block)}</div>')
    elif block.startswith("- "):
        items = [l[2:] for l in block.split("\n") if l.startswith("- ")]
        parts.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
    elif in_references:
        for line in block.split("\n"):
            parts.append(f'<p class="ref">{inline(line)}</p>')
    else:
        parts.append(f"<p>{inline(' '.join(block.split(chr(10))))}</p>")

title = re.search(r"^# (.+)$", text, flags=re.M).group(1)
css = """
@page { size: letter; margin: 1in; }
body { font-family: "Times New Roman", Times, serif; font-size: 12pt; line-height: 1.15; color: #000; background: #fff; }
h1 { font-size: 16pt; text-align: center; margin: 0 0 10pt; line-height: 1.25; }
h2 { font-size: 13pt; margin: 14pt 0 4pt; break-after: avoid; }
h3 { font-size: 12pt; margin: 10pt 0 3pt; break-after: avoid; }
h4 { font-size: 12pt; font-style: italic; font-weight: normal; margin: 8pt 0 2pt; break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; orphans: 2; widows: 2; }
p.authors { text-align: center; margin: 0 0 4pt; }
p.affil { text-align: center; font-size: 11pt; margin: 0 0 2pt; }
p.ref { text-align: left; padding-left: 1.5em; text-indent: -1.5em; margin: 0 0 3pt; font-size: 11pt; overflow-wrap: anywhere; }
ul { margin: 0 0 6pt 1.2em; padding: 0; } li { margin: 0 0 2pt; }
sup { font-size: 0.7em; line-height: 0; }
code { font-family: Menlo, Consolas, monospace; font-size: 10pt; }
figure { margin: 8pt 0 10pt; text-align: center; break-inside: avoid; }
figure img { display: block; margin: 0 auto 4pt; }
figcaption, .tcaption { font-size: 11pt; text-align: left; line-height: 1.2; }
.table-wrap { margin: 8pt 0 10pt; break-inside: avoid; }
.tcaption { margin-bottom: 4pt; }
table { border-collapse: collapse; width: 100%; font-size: 10.5pt; }
th, td { border-top: 0.75pt solid #000; border-bottom: 0.75pt solid #000; padding: 2pt 4pt; text-align: center; vertical-align: top; }
th { font-weight: bold; } td:first-child, th:first-child { text-align: left; }
"""
dst.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{css}</style></head>
<body>
{chr(10).join(parts)}
</body></html>
""")

import subprocess
pdf = HERE / (title.replace(":", " -") + ".pdf")
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                "--allow-file-access-from-files", f"--print-to-pdf={pdf}", dst.as_uri()],
               check=True, capture_output=True)
print("wrote", pdf)
