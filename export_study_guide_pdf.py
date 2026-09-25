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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — Enterprise Masterclass</title>

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">

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
            --border-color: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(56, 189, 248, 0.3);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --cyan-accent: #38bdf8;
            --indigo-accent: #818cf8;
            --emerald-accent: #34d399;
            --amber-accent: #fbbf24;
            --rose-accent: #f43f5e;
            --sidebar-width: 320px;
        }}

        [data-theme="light"] {{
            --bg-base: #f8fafc;
            --bg-surface: #ffffff;
            --bg-card: rgba(255, 255, 255, 0.9);
            --bg-hover: #f1f5f9;
            --border-color: #e2e8f0;
            --border-hover: #0284c7;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --cyan-accent: #0284c7;
            --indigo-accent: #4f46e5;
            --emerald-accent: #059669;
            --amber-accent: #d97706;
            --rose-accent: #e11d48;
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
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 28px;
            z-index: 1000;
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
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(129, 140, 248, 0.2));
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: var(--cyan-accent);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 3px 8px;
            border-radius: 6px;
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
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.6));
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 36px 40px;
            margin-bottom: 40px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
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

        .hero-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 34px;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #ffffff 40%, var(--cyan-accent) 100%);
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
            color: #ffffff;
            scroll-margin-top: 80px;
        }}

        .markdown-body h1 {{
            font-size: 28px;
            margin-top: 48px;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-color);
        }}

        .markdown-body h2 {{
            font-size: 22px;
            margin-top: 40px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            color: var(--cyan-accent);
        }}

        .markdown-body h3 {{
            font-size: 18px;
            margin-top: 28px;
            margin-bottom: 12px;
            color: #e2e8f0;
        }}

        .markdown-body h4 {{
            font-size: 15px;
            margin-top: 20px;
            margin-bottom: 8px;
            color: var(--indigo-accent);
        }}

        .markdown-body p {{
            margin-bottom: 16px;
            color: #cbd5e1;
        }}

        .markdown-body a {{
            color: var(--cyan-accent);
            text-decoration: none;
            border-bottom: 1px dashed rgba(56, 189, 248, 0.4);
            transition: all 0.15s;
        }}

        .markdown-body a:hover {{
            color: #7dd3fc;
            border-bottom-style: solid;
        }}

        .markdown-body ul, .markdown-body ol {{
            margin-bottom: 18px;
            padding-left: 24px;
            color: #cbd5e1;
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
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            background-color: var(--bg-surface);
            object-fit: contain;
        }}

        .markdown-body hr {{
            border: 0;
            height: 1px;
            background: var(--border-color);
            margin: 40px 0;
        }}

        /* Code Blocks & Copy Button */
        .code-container {{
            position: relative;
            margin: 20px 0;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            background: #0d1117;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}

        .code-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(22, 27, 34, 0.9);
            padding: 8px 16px;
            font-size: 12px;
            color: var(--text-muted);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-family: 'Fira Code', monospace;
        }}

        .code-header .lang-badge {{
            text-transform: uppercase;
            font-weight: 600;
            color: var(--cyan-accent);
        }}

        .copy-btn {{
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 11px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .copy-btn:hover {{
            background: var(--bg-hover);
            color: var(--cyan-accent);
            border-color: var(--cyan-accent);
        }}

        .markdown-body pre {{
            margin: 0;
            padding: 16px 20px;
            overflow-x: auto;
            font-family: 'Fira Code', monospace;
            font-size: 13.5px;
            line-height: 1.6;
            background: #0d1117 !important;
            color: #e6edf3;
        }}

        .markdown-body code {{
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
            background: rgba(255, 255, 255, 0.08);
            padding: 2px 6px;
            border-radius: 4px;
            color: #38bdf8;
        }}

        .markdown-body pre code {{
            background: transparent;
            padding: 0;
            color: inherit;
        }}

        /* Tables */
        .markdown-body table {{
            width: 100%;
            border-collapse: collapse;
            margin: 24px 0;
            font-size: 14px;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
        }}

        .markdown-body th {{
            background: rgba(30, 41, 59, 0.8);
            color: #ffffff;
            font-weight: 600;
            text-align: left;
            padding: 12px 16px;
            border-bottom: 2px solid var(--border-color);
        }}

        .markdown-body td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            color: #cbd5e1;
        }}

        .markdown-body tr:last-child td {{
            border-bottom: none;
        }}

        .markdown-body tr:hover td {{
            background: var(--bg-hover);
        }}

        /* Blockquotes as Callout Alerts */
        .markdown-body blockquote {{
            margin: 20px 0;
            padding: 16px 20px;
            background: rgba(30, 41, 59, 0.4);
            border-left: 4px solid var(--cyan-accent);
            border-radius: 0 8px 8px 0;
            color: #e2e8f0;
        }}

        /* Math Equations Card */
        .MathJax_Display, .mjx-chtml {{
            padding: 14px 0;
            overflow-x: auto;
            overflow-y: hidden;
        }}

        /* Mermaid Diagrams Container */
        .mermaid {{
            margin: 24px 0;
            padding: 20px;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            display: flex;
            justify-content: center;
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
                <li class="nav-item"><a href="#1-paper-reading-roadmap">Paper Reading Roadmap</a></li>
                <li class="nav-item"><a href="#2-newton-raphson-intuition-why-the-hessian-matters">Newton Intuition (Curvature)</a></li>
                <li class="nav-item"><a href="#3-complete-mathematical-derivation-of-objective--gain">Mathematical Derivations</a></li>
                <li class="nav-item"><a href="#7-the-4-core-diagnostic-questions-self-check">4 Diagnostic Questions</a></li>
                <li class="nav-item"><a href="#8-code-deliverables--c-parity-validation">Scratch Engine & Parity</a></li>
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
                        <span class="stat-value">17 / 17</span>
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

        // 2. Theme Toggle
        function toggleTheme() {{
            const current = document.documentElement.getAttribute('data-theme');
            const target = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', target);
        }}

        // 3. Search / TOC Filter
        function filterToc() {{
            const q = document.getElementById('toc-filter').value.toLowerCase();
            const items = document.querySelectorAll('#sidebar-toc .nav-item');
            items.forEach(item => {{
                const text = item.textContent.toLowerCase();
                item.style.display = text.includes(q) ? 'block' : 'none';
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
