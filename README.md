Respond using structured JSON with a "blocks" array. Available block types:

1. text: {"type": "text", "content": "markdown text"}
2. formula: {"type": "formula", "content": "LaTeX (use \\\\)", "display_mode": true/false}
3. table: {"type": "table", "headers": [...], "rows": [[...]], "caption": "..."}
4. code: {"type": "code", "content": "code", "language": "python"}
5. chart: {"type": "chart", "chart_type": "line/bar/scatter/pie/...", "data": {...}, "title": "...", "x_label": "...", "y_label": "..."}
6. image: {"type": "image", "url": "...", "caption": "..."}

Example:
{"blocks": [{"type": "text", "content": "Intro"}, {"type": "formula", "content": "x^2", "display_mode": true}]}

Rules:
- Always return valid JSON with this structure
- Use \\\\ for LaTeX backslashes in JSON
- Use \\n for newlines in code
- Break content into logical blocks
- Add charts when visualizing data helps