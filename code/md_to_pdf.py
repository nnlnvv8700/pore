# -*- coding: utf-8 -*-
"""
Convert README.md to PDF using weasyprint.
"""
import os
import markdown2

# Paths
md_path = r"E:\mhw\1\pore\code\README.md"
html_path = r"E:\mhw\1\pore\code\README.html"
pdf_path = r"E:\mhw\1\pore\code\README.pdf"

# Read markdown
with open(md_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# Convert to HTML
html_body = markdown2.markdown(
    md_content,
    extras=["fenced-code-blocks", "tables", "code-friendly", "cuddled-lists"]
)

# Full HTML with CSS styling
html_full = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>多孔介质流动场预测项目</title>
<style>
body {{
    font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    margin: 40px;
    color: #333;
}}
h1 {{
    color: #1a5276;
    border-bottom: 2px solid #1a5276;
    padding-bottom: 10px;
    font-size: 24pt;
}}
h2 {{
    color: #2874a6;
    border-bottom: 1px solid #aed6f1;
    padding-bottom: 5px;
    margin-top: 30px;
    font-size: 16pt;
}}
h3 {{
    color: #3498db;
    margin-top: 20px;
    font-size: 13pt;
}}
h4 {{
    color: #5dade2;
    font-size: 11pt;
}}
code {{
    background-color: #f4f6f7;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 10pt;
}}
pre {{
    background-color: #2c3e50;
    color: #ecf0f1;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.4;
}}
pre code {{
    background-color: transparent;
    padding: 0;
    color: #ecf0f1;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 10pt;
}}
th, td {{
    border: 1px solid #bdc3c7;
    padding: 8px 12px;
    text-align: left;
}}
th {{
    background-color: #3498db;
    color: white;
}}
tr:nth-child(even) {{
    background-color: #f8f9f9;
}}
blockquote {{
    border-left: 4px solid #3498db;
    padding-left: 15px;
    margin-left: 0;
    color: #555;
    font-style: italic;
}}
hr {{
    border: none;
    border-top: 1px solid #bdc3c7;
    margin: 30px 0;
}}
ul, ol {{
    margin-left: 20px;
}}
li {{
    margin-bottom: 5px;
}}
strong {{
    color: #1a5276;
}}
a {{
    color: #2980b9;
    text-decoration: none;
}}
.toc {{
    background-color: #f8f9f9;
    padding: 20px;
    border-radius: 5px;
    margin-bottom: 30px;
}}
@page {{
    margin: 2cm;
    @bottom-center {{
        content: counter(page);
    }}
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

# Save HTML
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_full)
print(f"Saved HTML: {html_path}")

# Convert to PDF
try:
    from weasyprint import HTML
    HTML(html_path).write_pdf(pdf_path)
    print(f"Saved PDF: {pdf_path}")
except Exception as e:
    print(f"PDF generation failed: {e}")
    print("HTML file has been created. You can open it in a browser and print to PDF.")
