import os
import sys
import webbrowser
import markdown

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <!-- MathJax for rendering LaTeX math formulas -->
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      }},
      svg: {{
        fontCache: 'global'
      }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 20mm 15mm 20mm 15mm;
            }}
            body {{
                background: #fff !important;
                color: #000 !important;
            }}
            .markdown-body pre {{
                border: 1px solid #ccc !important;
                background-color: #f8f8f8 !important;
                page-break-inside: avoid;
            }}
            .markdown-body h1, .markdown-body h2, .markdown-body h3 {{
                page-break-after: avoid;
            }}
        }}
        body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 960px;
            margin: 0 auto;
            padding: 35px 25px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
        }}
        .markdown-body {{
            font-size: 15px;
            line-height: 1.6;
            color: #c9d1d9;
            background-color: #0d1117;
        }}
        .markdown-body h1, .markdown-body h2 {{
            border-bottom-color: #21262d;
            color: #58a6ff;
        }}
        .markdown-body table {{
            display: table !important;
            width: 100% !important;
        }}
        .markdown-body pre {{
            background-color: #161b22 !important;
            border-radius: 6px;
            border: 1px solid #30363d;
        }}
        .print-btn {{
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #238636;
            color: white;
            padding: 10px 18px;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: 14px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 9999;
        }}
        .print-btn:hover {{
            background-color: #2ea043;
        }}
        @media print {{
            .print-btn {{ display: none !important; }}
            body, .markdown-body {{ background: white !important; color: black !important; }}
            .markdown-body h1, .markdown-body h2 {{ color: #000 !important; border-bottom-color: #ccc !important; }}
        }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ Save as PDF (Ctrl+P)</button>
    <article class="markdown-body">
        {content}
    </article>
</body>
</html>
"""

def generate_pdf_preview(md_file_path):
    if not os.path.exists(md_file_path):
        print(f"Error: Markdown file '{md_file_path}' not found.")
        sys.exit(1)

    print(f"Reading study guide: {md_file_path}")
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Convert Markdown to HTML
    html_content = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'codehilite', 'toc']
    )

    title = os.path.basename(md_file_path).replace('.md', '').replace('_', ' ')
    html_doc = HTML_TEMPLATE.format(title=title, content=html_content)

    output_html_path = md_file_path.replace('.md', '.html')
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_doc)

    abs_path = os.path.abspath(output_html_path)
    print(f"HTML Preview generated: {abs_path}")
    print("Opening in default web browser for PDF export (Ctrl+P -> Save as PDF)...")
    webbrowser.open(f"file:///{abs_path.replace('\\', '/')}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "XGBoost_Mastery_Study_Guide.md"
    generate_pdf_preview(target)
