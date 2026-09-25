"""
XGBoost Masterclass: Executive HTML & PDF Documentation Exporter
===============================================================

Transforms XGBoost_Mastery_Study_Guide.md into an enterprise-grade,
interactive documentation portal with:
- Sticky hierarchical navigation sidebar with real-time search filter and scrollspy
- Glassmorphic dark executive design with harmonious accents and Google Fonts
- Interactive code blocks with copy-to-clipboard buttons and language badges
- MathJax LaTeX mathematical equation rendering
- Mermaid.js native diagram rendering for architectures and mindmaps
- Executive stats dashboard and callout alert styling
- Print-optimized CSS (@media print) for pixel-perfect PDF export via Ctrl+P
"""

from __future__ import annotations

import os
import sys
import webbrowser

import markdown

HTML_PORTAL_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <script>
        (function() {{
            const saved = localStorage.getItem('study_guide_theme');
            if (saved) document.documentElement.setAttribute('data-theme', saved);
        }})();
    </script>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — Enterprise Masterclass</title>

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">

    <!-- MathJax for Mathematical LaTeX Equations -->
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      svg: {{ fontCache: 'global' }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <!-- Mermaid.js for Diagrams -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>

    <style>
        :root {{
            --bg-base: #080b12;
            --bg-surface: #0f172a;
            --bg-card: rgba(15, 23, 42, 0.75);
            --bg-hover: #1e293b;
            --navbar-bg: rgba(15, 23, 42, 0.85);
            --border-color: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(56, 189, 248, 0.3);
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --code-inline-bg: rgba(56, 189, 248, 0.1);
            --code-inline-color: #38bdf8;
            --code-inline-border: rgba(56, 189, 248, 0.2);
            --cyan-accent: #38bdf8;
            --indigo-accent: #818cf8;
            --emerald-accent: #34d399;
            --amber-accent: #fbbf24;
            --rose-accent: #f43f5e;
            --table-th-bg: #1e293b;
            --blockquote-bg: rgba(30, 41, 59, 0.5);
            --math-bg: #0d121f;
            --hero-bg: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.6));
            --hero-title-color: linear-gradient(135deg, #ffffff 40%, #38bdf8 100%);
            --hero-border: rgba(255, 255, 255, 0.08);
            --card-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            --sidebar-width: 320px;
        }}

        [data-theme="light"] {{
            --bg-base: #f4f0e6;             /* Warm drafting desk / book canvas */
            --bg-surface: #fdfcf7;          /* High-grade ivory archival publication paper */
            --bg-card: #f8f5ec;             /* Subtle warm inset card */
            --bg-hover: #ede7d8;            /* Warm parchment hover */
            --navbar-bg: rgba(253, 252, 247, 0.95);
            --border-color: #e2dcd0;        /* Fine bookbinding rule */
            --border-hover: #0f4c81;
            --text-primary: #121826;        /* Deep carbon printer's ink black */
            --text-secondary: #2b2824;      /* Editorial reading ink (warm charcoal) */
            --text-muted: #6e675f;          /* Graphite secondary annotation */
            --code-inline-bg: #f2ede0;      /* Warm cream chip */
            --code-inline-color: #8c281f;   /* Classic archival rust notation */
            --code-inline-border: #ded7c6;
            --cyan-accent: #0f4c81;         /* Oxford scholar blue */
            --indigo-accent: #1e3a8a;       /* Deep monograph navy */
            --emerald-accent: #14532d;      /* Deep forest green */
            --amber-accent: #92400e;        /* Dark warm amber */
            --rose-accent: #991b1b;         /* Crimson editorial mark */
            --table-th-bg: #f5f0e4;
            --blockquote-bg: #f7f3ea;
            --math-bg: #faf7ee;
            --hero-bg: #fbf9f2;
            --hero-title-color: #121826;
            --hero-border: #ded7c6;
            --card-shadow: 0 1px 3px rgba(40, 30, 20, 0.04), 0 8px 24px -4px rgba(40, 30, 20, 0.06);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            line-height: 1.7;
            font-size: 15.5px;
            scroll-behavior: smooth;
            background-image:
                radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 60%, rgba(129, 140, 248, 0.06) 0%, transparent 40%);
            background-attachment: fixed;
        }}

        [data-theme="light"] body {{
            background-image:
                radial-gradient(circle at 15% 10%, rgba(2, 132, 199, 0.04) 0%, transparent 40%),
                radial-gradient(circle at 85% 60%, rgba(79, 70, 229, 0.03) 0%, transparent 40%);
        }}

        /* Reading Progress Bar */
        #progress-bar {{
            position: fixed;
            top: 0;
            left: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--cyan-accent), var(--indigo-accent), var(--emerald-accent));
            width: 0%;
            z-index: 10000;
            transition: width 0.1s ease;
        }}

        /* Top Header Bar */
        .top-navbar {{
            position: sticky;
            top: 0;
            left: 0;
            right: 0;
            height: 64px;
            background: var(--navbar-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 28px;
            z-index: 1000;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        }}

        .nav-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'Outfit', sans-serif;
            font-size: 17px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-primary);
        }}

        .brand-badge {{
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: var(--cyan-accent);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 3px 8px;
            border-radius: 6px;
        }}

        [data-theme="light"] .brand-badge {{
            background: rgba(2, 132, 199, 0.1);
            border-color: rgba(2, 132, 199, 0.25);
            color: #0284c7;
        }}

        .nav-actions {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .action-btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
        }}

        .action-btn:hover {{
            background: var(--bg-hover);
            border-color: var(--border-hover);
            color: var(--cyan-accent);
            transform: translateY(-1px);
        }}

        .btn-primary {{
            background: linear-gradient(135deg, #0284c7, #2563eb);
            border: none;
            color: white;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
        }}

        .btn-primary:hover {{
            background: linear-gradient(135deg, #0369a1, #1d4ed8);
            color: white;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5);
        }}

        /* Master Container Layout */
        .portal-layout {{
            display: flex;
            max-width: 1600px;
            margin: 0 auto;
            min-height: calc(100vh - 64px);
        }}

        /* Sidebar Navigation */
        .portal-sidebar {{
            width: var(--sidebar-width);
            flex-shrink: 0;
            position: sticky;
            top: 64px;
            height: calc(100vh - 64px);
            overflow-y: auto;
            padding: 24px 18px 40px 24px;
            border-right: 1px solid var(--border-color);
            background: var(--bg-surface);
        }}

        .portal-sidebar::-webkit-scrollbar {{
            width: 5px;
        }}
        .portal-sidebar::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 4px;
        }}

        .sidebar-search {{
            position: relative;
            margin-bottom: 20px;
        }}

        .sidebar-search input {{
            width: 100%;
            background: var(--bg-base);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 9px 12px 9px 34px;
            border-radius: 8px;
            font-size: 13px;
            outline: none;
            transition: border-color 0.2s;
        }}

        .sidebar-search input:focus {{
            border-color: var(--cyan-accent);
        }}

        .sidebar-search svg {{
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            width: 15px;
            height: 15px;
            color: var(--text-muted);
        }}

        .nav-section-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            font-weight: 700;
            margin: 20px 0 8px 10px;
        }}

        .nav-links {{
            list-style: none;
        }}

        .nav-item {{
            margin-bottom: 2px;
        }}

        .nav-item a {{
            display: flex;
            align-items: center;
            padding: 7px 12px;
            color: var(--text-secondary);
            font-size: 13.5px;
            font-weight: 500;
            text-decoration: none;
            border-radius: 6px;
            transition: all 0.15s ease;
        }}

        .nav-item a:hover {{
            background: var(--bg-hover);
            color: var(--cyan-accent);
            padding-left: 16px;
        }}

        .nav-item a.active {{
            background: rgba(56, 189, 248, 0.12);
            color: var(--cyan-accent);
            font-weight: 600;
            border-left: 3px solid var(--cyan-accent);
            border-radius: 0 6px 6px 0;
        }}

        /* Main Content Pane */
        .portal-main {{
            flex: 1;
            min-width: 0;
            max-width: calc(100% - var(--sidebar-width));
            padding: 40px 60px 80px;
            overflow-x: hidden;
        }}

        /* Hero Executive Dashboard */
        .hero-banner {{
            background: var(--hero-bg);
            border: 1px solid var(--hero-border);
            border-radius: 16px;
            padding: 36px 40px;
            margin-bottom: 40px;
            position: relative;
            overflow: hidden;
            box-shadow: var(--card-shadow);
        }}

        .hero-banner::after {{
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 300px;
            height: 100%;
            background: radial-gradient(circle at top right, rgba(56, 189, 248, 0.15), transparent 70%);
            pointer-events: none;
        }}

        [data-theme="light"] .hero-banner::after {{
            background: radial-gradient(circle at top right, rgba(2, 132, 199, 0.08), transparent 70%);
        }}

        .hero-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 34px;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: var(--hero-title-color);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }}

        .hero-subtitle {{
            color: var(--text-secondary);
            font-size: 15px;
            max-width: 800px;
            line-height: 1.6;
            margin-bottom: 24px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }}

        .stat-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px 18px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        }}

        .stat-value {{
            font-family: 'Outfit', sans-serif;
            font-size: 24px;
            font-weight: 700;
            color: var(--cyan-accent);
        }}

        .stat-label {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-top: 2px;
        }}

        /* Markdown Typography & Elements */
        .markdown-body {{
            width: 100%;
            max-width: 100%;
            overflow-x: hidden;
            color: var(--text-primary);
        }}

        .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {{
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.02em;
            font-weight: 700;
            color: var(--text-primary);
            scroll-margin-top: 80px;
        }}

        .markdown-body h1 {{
            font-size: 28px;
            margin-top: 48px;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}

        .markdown-body h2 {{
            font-size: 22px;
            margin-top: 40px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
            color: var(--cyan-accent);
        }}

        .markdown-body h3 {{
            font-size: 18px;
            margin-top: 28px;
            margin-bottom: 12px;
            color: var(--text-primary);
        }}

        .markdown-body h4 {{
            font-size: 15px;
            margin-top: 20px;
            margin-bottom: 8px;
            color: var(--indigo-accent);
        }}

        .markdown-body p {{
            margin-bottom: 16px;
            color: var(--text-secondary);
            font-size: 15.5px;
            line-height: 1.75;
        }}

        .markdown-body strong, .markdown-body b {{
            color: var(--text-primary);
            font-weight: 600;
        }}

        .markdown-body a {{
            color: var(--cyan-accent);
            text-decoration: none;
            border-bottom: 1px dashed var(--cyan-accent);
            transition: all 0.15s;
        }}

        .markdown-body a:hover {{
            color: var(--indigo-accent);
            border-bottom-style: solid;
        }}

        .markdown-body ul, .markdown-body ol {{
            margin-bottom: 18px;
            padding-left: 24px;
            color: var(--text-secondary);
            line-height: 1.75;
        }}

        .markdown-body li {{
            margin-bottom: 6px;
        }}

        .markdown-body img, img {{
            max-width: 100% !important;
            height: auto !important;
            display: block;
            margin: 28px auto;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            background-color: var(--bg-surface);
            object-fit: contain;
        }}

        .markdown-body hr {{
            border: 0;
            height: 1px;
            background: var(--border-color);
            margin: 40px 0;
        }}

        /* Code Blocks & Copy Button - Always Dark High-Contrast Terminal */
        .code-container {{
            position: relative;
            margin: 22px 0;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #30363d;
            background: #0d1117;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
        }}

        .code-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #161b22;
            padding: 8px 16px;
            font-size: 12px;
            color: #8b949e;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            font-family: 'Fira Code', monospace;
        }}

        .code-header .lang-badge {{
            text-transform: uppercase;
            font-weight: 700;
            color: #58a6ff;
            font-size: 11px;
            letter-spacing: 0.05em;
        }}

        .copy-btn {{
            background: rgba(240, 246, 252, 0.08);
            border: 1px solid rgba(240, 246, 252, 0.15);
            color: #c9d1d9;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .copy-btn:hover {{
            background: rgba(88, 166, 255, 0.15);
            color: #58a6ff;
            border-color: #58a6ff;
        }}

        .markdown-body pre {{
            margin: 0;
            padding: 16px 20px;
            overflow-x: auto;
            font-family: 'Fira Code', monospace;
            font-size: 13.5px;
            line-height: 1.6;
            background: #0d1117 !important;
            color: #e6edf3 !important;
        }}

        .markdown-body code {{
            font-family: 'Fira Code', monospace;
            font-size: 0.88em;
            background: var(--code-inline-bg);
            border: 1px solid var(--code-inline-border);
            padding: 2px 6px;
            border-radius: 5px;
            color: var(--code-inline-color);
            font-weight: 500;
        }}

        .markdown-body pre code {{
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            color: inherit !important;
        }}

        /* Tables */
        .markdown-body table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 24px 0;
            font-size: 14px;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }}

        .markdown-body th {{
            background: var(--table-th-bg);
            color: var(--text-primary);
            font-weight: 700;
            text-align: left;
            padding: 12px 16px;
            border-bottom: 2px solid var(--border-color);
        }}

        .markdown-body td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-secondary);
        }}

        .markdown-body tr:last-child td {{
            border-bottom: none;
        }}

        .markdown-body tr:hover td {{
            background: var(--bg-hover);
        }}

        /* Blockquotes as Callout Alerts */
        .markdown-body blockquote {{
            margin: 22px 0;
            padding: 16px 22px;
            background: var(--blockquote-bg);
            border-left: 4px solid var(--cyan-accent);
            border-top: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            border-bottom: 1px solid var(--border-color);
            border-radius: 0 10px 10px 0;
            color: var(--text-secondary);
            font-style: normal;
        }}

        .markdown-body blockquote p:last-child {{
            margin-bottom: 0;
        }}

        /* Math Equations Display */
        .MathJax, .mjx-chtml, .MathJax_Display {{
            color: var(--text-primary) !important;
        }}

        .MathJax_Display {{
            padding: 16px 20px !important;
            margin: 20px 0 !important;
            background: var(--math-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            overflow-x: auto;
            overflow-y: hidden;
        }}

        /* Mermaid Diagrams Container */
        .mermaid {{
            margin: 24px 0;
            padding: 24px;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            display: flex;
            justify-content: center;
        }}

        /* ======================================================== */
        /* ACADEMIC PAPER MONOGRAPH AESTHETIC (LIGHT THEME)         */
        /* ======================================================== */
        [data-theme="light"] body {{
            background-color: var(--bg-base);
            color: var(--text-secondary);
            background-image:
                linear-gradient(rgba(18, 24, 38, 0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(18, 24, 38, 0.025) 1px, transparent 1px);
            background-size: 28px 28px;
        }}

        [data-theme="light"] .top-navbar {{
            background: rgba(253, 252, 247, 0.95);
            border-bottom: 1px solid #ded8cb;
            box-shadow: 0 1px 4px rgba(40, 30, 20, 0.04);
        }}

        [data-theme="light"] .nav-brand {{
            font-family: 'Newsreader', 'Georgia', serif;
            font-size: 19px;
            font-weight: 700;
            color: #121826;
            letter-spacing: -0.01em;
        }}

        [data-theme="light"] .brand-badge {{
            background: #f0ebe0;
            border: 1px solid #ded7c6;
            color: #0f4c81;
            font-family: 'Inter', sans-serif;
            font-size: 10.5px;
        }}

        [data-theme="light"] .portal-sidebar {{
            background: #faf7f0;
            border-right: 1px solid #e2dcd0;
        }}

        [data-theme="light"] .sidebar-search input {{
            background: #fdfcf7;
            border: 1px solid #ded7c6;
            color: #121826;
        }}

        [data-theme="light"] .nav-section-title {{
            color: #78716a;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.12em;
        }}

        [data-theme="light"] .nav-item a {{
            color: #4b4640;
            font-family: 'Inter', sans-serif;
            font-size: 13.5px;
        }}

        [data-theme="light"] .nav-item a:hover {{
            background: #f0ebe0;
            color: #0f4c81;
        }}

        [data-theme="light"] .nav-item a.active {{
            background: #eae4d6;
            color: #0f4c81;
            border-left: 3px solid #0f4c81;
            font-weight: 600;
        }}

        /* Paper sheet monograph container */
        [data-theme="light"] .portal-main {{
            background: var(--bg-surface);
            max-width: 1060px;
            margin: 32px auto 60px;
            padding: 56px 76px 110px;
            border-radius: 4px;
            border: 1px solid #ded7c6;
            box-shadow:
                0 1px 3px rgba(40, 30, 20, 0.04),
                0 12px 36px -4px rgba(40, 30, 20, 0.08);
        }}

        [data-theme="light"] .hero-banner {{
            background: #fbf9f2;
            border: 1px solid #ded7c6;
            border-top: 4px solid #0f4c81;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(40, 30, 20, 0.03);
            padding: 36px 42px;
        }}

        [data-theme="light"] .hero-banner::after {{
            display: none;
        }}

        [data-theme="light"] .hero-title {{
            font-family: 'Newsreader', 'Georgia', serif;
            font-size: 35px;
            font-weight: 700;
            letter-spacing: -0.015em;
            background: none;
            -webkit-text-fill-color: #121826;
            color: #121826;
        }}

        [data-theme="light"] .hero-subtitle {{
            font-family: 'Inter', sans-serif;
            color: #4b4640;
            font-size: 15px;
            line-height: 1.65;
        }}

        [data-theme="light"] .stat-card {{
            background: #fdfcf7;
            border: 1px solid #e3ded2;
            border-radius: 4px;
            box-shadow: none;
        }}

        [data-theme="light"] .stat-value {{
            color: #0f4c81;
            font-family: 'Newsreader', 'Georgia', serif;
            font-weight: 700;
            font-size: 26px;
        }}

        /* Publication Typography */
        [data-theme="light"] .markdown-body {{
            font-family: 'Newsreader', 'Charter', 'Georgia', serif;
            font-size: 16.5px;
            line-height: 1.84;
            color: var(--text-secondary);
        }}

        [data-theme="light"] .markdown-body h1,
        [data-theme="light"] .markdown-body h2,
        [data-theme="light"] .markdown-body h3,
        [data-theme="light"] .markdown-body h4 {{
            font-family: 'Newsreader', 'Georgia', serif;
            color: var(--text-primary);
            letter-spacing: -0.015em;
        }}

        [data-theme="light"] .markdown-body h1 {{
            font-size: 32px;
            font-weight: 700;
            border-bottom: 2px solid #121826;
            padding-bottom: 14px;
            margin-top: 56px;
            margin-bottom: 22px;
        }}

        [data-theme="light"] .markdown-body h2 {{
            font-size: 24px;
            font-weight: 700;
            color: #0f4c81;
            border-bottom: 1px solid #ded7c6;
            padding-bottom: 8px;
            margin-top: 44px;
            margin-bottom: 18px;
        }}

        [data-theme="light"] .markdown-body h3 {{
            font-size: 19.5px;
            font-weight: 600;
            color: #1f2937;
            margin-top: 32px;
            margin-bottom: 14px;
        }}

        [data-theme="light"] .markdown-body h4 {{
            font-size: 16px;
            font-weight: 600;
            color: #0f4c81;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        [data-theme="light"] .markdown-body p {{
            color: #2b2824;
            margin-bottom: 18px;
        }}

        [data-theme="light"] .markdown-body strong,
        [data-theme="light"] .markdown-body b {{
            color: #121826;
            font-weight: 700;
        }}

        [data-theme="light"] .markdown-body ul,
        [data-theme="light"] .markdown-body ol {{
            color: #2b2824;
            line-height: 1.82;
            margin-bottom: 20px;
        }}

        [data-theme="light"] .markdown-body a {{
            color: #0f4c81;
            border-bottom: 1px solid #0f4c81;
        }}

        [data-theme="light"] .markdown-body a:hover {{
            color: #1e3a8a;
            border-bottom: 1.5px solid #1e3a8a;
        }}

        /* Formal Academic Table (IEEE / Nature / Chicago Manual format) */
        [data-theme="light"] .markdown-body table {{
            border-top: 2.5px solid #121826;
            border-bottom: 2.5px solid #121826;
            border-left: none;
            border-right: none;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
            margin: 28px 0;
        }}

        [data-theme="light"] .markdown-body th {{
            border-bottom: 1.5px solid #121826;
            background: #f7f3e8;
            color: #121826;
            font-family: 'Newsreader', 'Georgia', serif;
            font-weight: 700;
            font-size: 15px;
            letter-spacing: 0.02em;
            padding: 10px 14px;
        }}

        [data-theme="light"] .markdown-body td {{
            border-bottom: 1px solid #e7e1d4;
            color: #2b2824;
            font-size: 14.5px;
            font-family: 'Inter', -apple-system, sans-serif;
            padding: 11px 14px;
        }}

        [data-theme="light"] .markdown-body tr:hover td {{
            background: #f7f4ea;
        }}

        /* Academic Theorem / Pull-quote Callout */
        [data-theme="light"] .markdown-body blockquote {{
            background: #f8f5ed;
            border-left: 3.5px solid #0f4c81;
            border-top: 1px solid #e8e2d4;
            border-right: 1px solid #e8e2d4;
            border-bottom: 1px solid #e8e2d4;
            border-radius: 0 4px 4px 0;
            color: #38342e;
            font-style: italic;
            padding: 18px 24px;
            margin: 24px 0;
        }}

        /* TeX / MathJax Display Paper Inset */
        [data-theme="light"] .MathJax,
        [data-theme="light"] .mjx-chtml,
        [data-theme="light"] .MathJax_Display {{
            color: #121826 !important;
        }}

        [data-theme="light"] .MathJax_Display {{
            background: #faf7ee;
            border: 1px solid #e2dbcd;
            border-radius: 4px;
            box-shadow: inset 0 1px 2px rgba(40, 30, 20, 0.03);
            padding: 20px 24px !important;
            margin: 24px 0 !important;
        }}

        /* Monograph Code Containers */
        [data-theme="light"] .code-container {{
            border: 1px solid #333a42;
            box-shadow: 0 4px 16px rgba(40, 30, 20, 0.12);
            border-radius: 6px;
        }}

        [data-theme="light"] .markdown-body code {{
            background: #f2ede0;
            border: 1px solid #ded7c6;
            color: #8c281f;
            font-weight: 600;
        }}

        /* Print Optimization */
        @media print {{
            #progress-bar, .top-navbar, .portal-sidebar, .copy-btn {{
                display: none !important;
            }}
            body {{
                background: #ffffff !important;
                color: #000000 !important;
                background-image: none !important;
            }}
            .portal-layout {{
                display: block !important;
            }}
            .portal-main {{
                padding: 0 !important;
                max-width: 100% !important;
            }}
            .hero-banner {{
                border: 1px solid #ccc !important;
                background: #f8fafc !important;
                box-shadow: none !important;
            }}
            .hero-title {{
                -webkit-text-fill-color: #000000 !important;
            }}
            .markdown-body h1, .markdown-body h2, .markdown-body h3 {{
                color: #000000 !important;
                page-break-after: avoid;
            }}
            .code-container, pre {{
                background: #f8fafc !important;
                color: #000000 !important;
                border: 1px solid #ccc !important;
                page-break-inside: avoid;
            }}
            table {{
                border: 1px solid #ccc !important;
            }}
            th {{
                background: #f1f5f9 !important;
                color: #000000 !important;
            }}
            td {{
                color: #000000 !important;
            }}
        }}

        /* Responsive Layout */
        @media (max-width: 1024px) {{
            .portal-sidebar {{
                display: none;
            }}
            .portal-main {{
                padding: 24px 20px;
            }}
        }}
    </style>
</head>
<body>
    <!-- Reading Progress Bar -->
    <div id="progress-bar"></div>

    <!-- Top Navigation Header -->
    <header class="top-navbar">
        <div class="nav-brand">
            <span>⚡ XGBoost Mastery</span>
            <span class="brand-badge">Enterprise Edition</span>
        </div>
        <div class="nav-actions">
            <button class="action-btn" onclick="toggleTheme()" title="Toggle Dark/Light Mode">🌓 Theme</button>
            <button class="action-btn btn-primary" onclick="window.print()">🖨️ Save as PDF / Print</button>
        </div>
    </header>

    <!-- Master Layout -->
    <div class="portal-layout">
        <!-- Sidebar Navigation -->
        <aside class="portal-sidebar">
            <div class="sidebar-search">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                <input type="text" id="toc-filter" placeholder="Filter sections... (Ctrl+K)" onkeyup="filterToc()">
            </div>

            <div class="nav-section-title">Core Curriculum</div>
            <ul class="nav-links" id="sidebar-toc">
                <li class="nav-item"><a href="#module-01-theoretical-foundations">01. Theoretical Foundations</a></li>
                <li class="nav-item"><a href="#module-02-xgboost-core-mechanics">02. XGBoost Core Mechanics</a></li>
                <li class="nav-item"><a href="#module-03-basic-usage--tuning">03. Tuning & Optuna HPO</a></li>
                <li class="nav-item"><a href="#module-04-advanced-features--diagnostics">04. Enterprise Interpretability</a></li>
                <li class="nav-item"><a href="#module-05-production-deployment--serving">05. Serving & Governance</a></li>
                <li class="nav-item"><a href="#module-06-banking--finance-projects">06. Banking & Quant Capstones</a></li>
                <li class="nav-item"><a href="#module-07-distributed-xgboost-architecture">07. Distributed Architecture</a></li>
            </ul>

            <div class="nav-section-title">Core Engine Quicklinks</div>
            <ul class="nav-links">
                <li class="nav-item"><a href="#1-theoretical-architecture-formulation-chen-guestrin-2016">Theoretical Pillars & Formulation</a></li>
                <li class="nav-item"><a href="#2-newton-raphson-intuition-why-the-hessian-matters">Newton Intuition (Curvature)</a></li>
                <li class="nav-item"><a href="#3-complete-mathematical-derivation-of-objective-gain">Mathematical Derivations</a></li>
                <li class="nav-item"><a href="#4-complete-worked-numerical-trace-the-pen-and-paper-proof">Worked Numerical Trace</a></li>
                <li class="nav-item"><a href="#6-the-approximate-split-engine-weighted-quantile-sketch-full-derivation">Weighted Quantile Sketch</a></li>
                <li class="nav-item"><a href="#7-systems-engineering-architecture-c-kernels">C++ Systems & Kernels</a></li>
                <li class="nav-item"><a href="#8-the-4-core-diagnostic-questions-self-check">4 Diagnostic Questions</a></li>
                <li class="nav-item"><a href="#9-code-deliverables-c-parity-validation">Scratch Engine & Parity</a></li>
            </ul>

            <div class="nav-section-title">Quantitative Risk & Uplift</div>
            <ul class="nav-links">
                <li class="nav-item"><a href="#10-conformal-prediction-distribution-free-uncertainty-guarantees">Conformal Risk Calibration</a></li>
                <li class="nav-item"><a href="#102-institutional-tripartite-underwriting-triage">Tripartite Underwriting Triage</a></li>
                <li class="nav-item"><a href="#103-conformalized-quantile-regression-cqr-on-heteroskedastic-loss">CQR Heteroskedastic Loss</a></li>
                <li class="nav-item"><a href="#104-mondrian-group-conditional-conformal-fairness-auditing">Mondrian Fairness Auditing</a></li>
                <li class="nav-item"><a href="#43-the-meta-learner-hierarchy-for-cate-estimation">Causal Uplift & X-Learner</a></li>
                <li class="nav-item"><a href="#45-closed-form-net-expected-value-nev-budget-policy-optimization">NEV Budget Policy Optimizer</a></li>
            </ul>
        </aside>

        <!-- Main Content Area -->
        <main class="portal-main">
            <!-- Hero Dashboard Banner -->
            <div class="hero-banner">
                <div class="hero-title">XGBoost Mastery: Theory to Production</div>
                <div class="hero-subtitle">
                    Comprehensive technical manual covering 2nd-order optimization derivations, pure NumPy from-scratch engines, enterprise banking capstones, and distributed scale.
                </div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-value">7</span>
                        <span class="stat-label">Core Modules</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">6</span>
                        <span class="stat-label">Finance Capstones</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">&lt; 4e-8</span>
                        <span class="stat-label">C++ Parity Error</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">30 / 30</span>
                        <span class="stat-label">Pytest Passed</span>
                    </div>
                </div>
            </div>

            <!-- Markdown Content Rendered -->
            <article class="markdown-body" id="doc-content">
                {content}
            </article>
        </main>
    </div>

    <!-- Client-Side Enhancements Script -->
    <script>
        // 1. Reading Progress Bar
        window.addEventListener('scroll', () => {{
            const winScroll = document.documentElement.scrollTop;
            const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            const scrolled = (winScroll / height) * 100;
            document.getElementById('progress-bar').style.width = scrolled + '%';
        }});

        // 2. Theme Persistence & Toggle
        const savedTheme = localStorage.getItem('study_guide_theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);

        function toggleTheme() {{
            const current = document.documentElement.getAttribute('data-theme');
            const target = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', target);
            localStorage.setItem('study_guide_theme', target);
        }}

        // 3. Search / TOC Filter
        function filterToc() {{
            const q = document.getElementById('toc-filter').value.toLowerCase();
            const items = document.querySelectorAll('.portal-sidebar .nav-item');
            items.forEach(item => {{
                const text = item.textContent.toLowerCase();
                item.style.display = text.includes(q) ? 'block' : 'none';
            }});
            document.querySelectorAll('.portal-sidebar .nav-section-title').forEach(title => {{
                const list = title.nextElementSibling;
                if (list && list.classList.contains('nav-links')) {{
                    const visibleItems = Array.from(list.querySelectorAll('.nav-item')).some(i => i.style.display !== 'none');
                    title.style.display = visibleItems ? 'block' : 'none';
                }}
            }});
        }}

        // Quick shortcut Ctrl+K to search
        window.addEventListener('keydown', (e) => {{
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {{
                e.preventDefault();
                const input = document.getElementById('toc-filter');
                if (input) input.focus();
            }}
        }});

        // 4. Wrap Code Blocks with Header and Copy Button
        document.addEventListener('DOMContentLoaded', () => {{
            document.querySelectorAll('pre > code').forEach((codeBlock) => {{
                const pre = codeBlock.parentElement;
                if (pre.parentElement.classList.contains('code-container')) return;

                // Detect language from class (e.g. language-python)
                let lang = 'CODE';
                codeBlock.classList.forEach(cls => {{
                    if (cls.startsWith('language-')) {{
                        lang = cls.replace('language-', '').toUpperCase();
                    }}
                }});

                const container = document.createElement('div');
                container.className = 'code-container';

                const header = document.createElement('div');
                header.className = 'code-header';
                header.innerHTML = `
                    <span class="lang-badge">${{lang}}</span>
                    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
                `;

                pre.parentNode.insertBefore(container, pre);
                container.appendChild(header);
                container.appendChild(pre);
            }});

            // 5. Convert Mermaid blocks to rendered diagrams
            document.querySelectorAll('pre code.language-mermaid').forEach(el => {{
                const parent = el.parentElement;
                const container = parent.parentElement;
                const div = document.createElement('div');
                div.className = 'mermaid';
                div.textContent = el.textContent;
                if (container && container.classList.contains('code-container')) {{
                    container.replaceWith(div);
                }} else {{
                    parent.replaceWith(div);
                }}
            }});
            // 6. Ensure all images are responsive and bounded
            document.querySelectorAll('img').forEach(img => {{
                img.style.maxWidth = '100%';
                img.style.height = 'auto';
                img.style.display = 'block';
            }});
        }});

        // Copy Code Functionality
        function copyCode(btn) {{
            const pre = btn.closest('.code-container').querySelector('pre');
            const text = pre.innerText;
            navigator.clipboard.writeText(text).then(() => {{
                btn.textContent = '✓ Copied!';
                btn.style.color = '#34d399';
                btn.style.borderColor = '#34d399';
                setTimeout(() => {{
                    btn.textContent = 'Copy';
                    btn.style.color = '';
                    btn.style.borderColor = '';
                }}, 2000);
            }});
        }}
    </script>
</body>
</html>
"""


def generate_pdf_preview(md_file_path: str) -> None:
    if not os.path.exists(md_file_path):
        print(f"Error: Markdown file '{md_file_path}' not found.")
        sys.exit(1)

    print(f"Reading study guide: {md_file_path}")
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Math and code fence protection pipeline:
    # Python-Markdown interprets underscores (_) and asterisks (*) as italics/bold,
    # which mutilates LaTeX subscripts (e.g., F_{m-1}(x), w_j^*, y_i - \bar{y}).
    # We substitute all math into unique alphanumeric placeholders before markdown rendering,
    # then restore them verbatim for MathJax to render cleanly.
    import re

    # Step 1: Protect code blocks so math inside code (or code ticks inside math) don't conflict
    code_cache = {}
    code_idx = 0
    def cache_code(m):
        nonlocal code_idx
        token = f"XYZCODEBLOCKTOKEN{code_idx}XYZ"
        code_idx += 1
        code_cache[token] = m.group(0)
        return token

    processed_md = re.sub(r"```[\s\S]*?```", cache_code, md_text)
    processed_md = re.sub(r"`[^`\n]+`", cache_code, processed_md)

    # Step 2: Protect display math $$...$$
    math_cache = {}
    math_idx = 0
    def cache_display_math(m):
        nonlocal math_idx
        token = f"XYZDISPLAYMATHTOKEN{math_idx}XYZ"
        math_idx += 1
        math_cache[token] = f"$${m.group(1)}$$"
        return token

    processed_md = re.sub(r"\$\$([\s\S]*?)\$\$", cache_display_math, processed_md)

    # Step 3: Protect inline math $...$
    def cache_inline_math(m):
        nonlocal math_idx
        token = f"XYZINLINEMATHTOKEN{math_idx}XYZ"
        math_idx += 1
        math_cache[token] = f"${m.group(1)}$"
        return token

    processed_md = re.sub(r"(?<!\\)\$(?!\s)(.+?)(?<!\s)(?<!\\)\$", cache_inline_math, processed_md)

    # Step 4: Restore code blocks before mermaid & markdown processing
    for token, code in code_cache.items():
        processed_md = processed_md.replace(token, code)

    # Step 5: Pre-process mermaid code fences so they are output directly as <div class="mermaid">
    processed_md = re.sub(
        r"```mermaid\s*\n(.*?)\n```",
        r'<div class="mermaid">\n\1\n</div>',
        processed_md,
        flags=re.DOTALL,
    )

    # Step 6: Convert Markdown to HTML with rich extensions
    html_content = markdown.markdown(
        processed_md,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "attr_list",
            "def_list",
            "abbr",
        ],
    )

    # Step 7: Restore all pristine LaTeX math blocks for MathJax
    for token, math in math_cache.items():
        html_content = html_content.replace(token, math)

    title = os.path.basename(md_file_path).replace(".md", "").replace("_", " ")
    html_doc = HTML_PORTAL_TEMPLATE.format(title=title, content=html_content)

    output_html_path = md_file_path.replace(".md", ".html")
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    abs_path = os.path.abspath(output_html_path)
    print(f"Enterprise HTML Portal generated: {output_html_path}")
    print("Opening in default web browser for interactive review or PDF export (Ctrl+P)...")
    url_path = abs_path.replace("\\", "/")
    webbrowser.open(f"file:///{url_path}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "XGBoost_Mastery_Study_Guide.md"
    generate_pdf_preview(target)
