# MIT License (c)2025 Andre Galberto - see LICENSE.md for full text
"""
Visualize p53 R248W mutation with py3Dmol.
"""

# Optional import of py3Dmol: don't fail module import if py3Dmol is not installed.
HAS_PY3DMOL = True
try:
 import py3Dmol
except Exception:
 HAS_PY3DMOL = False

import tempfile
import os
import urllib.request
from pathlib import Path
import json
import webbrowser


def export_html_view(pdb_id, pdb_text, chain='A', resi=248, mutation='R248W', width=800, height=600,
 outdir=None, open_in_browser=True):
 """Export a standalone HTML file with embedded3Dmol.js viewer showing the PDB.

 The HTML uses the3Dmol.js CDN and embeds the PDB text directly so it can be opened locally
 in a browser without a server. Returns the path to the generated HTML file.
 """
 if outdir is None:
 project_dir = Path(__file__).resolve().parent
 outdir = project_dir / 'pdbs'
 else:
 outdir = Path(outdir)
 outdir.mkdir(parents=True, exist_ok=True)

 safe_id = str(pdb_id).lower()
 html_path = outdir / f"{safe_id}_{resi}.html"

 # Use JSON encoding to safely embed the PDB text into JS string
 pdb_js_string = json.dumps(pdb_text)

 html = f"""<!doctype html>
 <html>
 <head>
 <meta charset='utf-8'/>
 <title>3D View - {safe_id} res {resi}</title>
 <script src='https://3dmol.csb.pitt.edu/build/3Dmol-min.js'></script>
 <style>body {{ margin:0; padding:0; }} #viewer {{ width:100%; height:100vh; }}</style>
 </head>
 <body>
 <div id='viewer'></div>
 <script>
 var pdbData = {pdb_js_string};
 var viewer = $3Dmol.createViewer('viewer', {{backgroundColor: 'white'}});
 viewer.addModel(pdbData, 'pdb');
 viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
 // highlight residue
 try {{
 viewer.addStyle({{chain: '{chain}', resi: {resi}}}, {{stick: {{colorscheme: 'magentaCarbon'}}, sphere: {{radius:1.0, color:'magenta'}}}});
 viewer.zoomTo({{chain: '{chain}', resi: {resi}}});
 }} catch(e) {{ /* ignore if selection fails */ }}
 viewer.render();
 </script>
 </body>
 </html>"""

 with open(html_path, 'w', encoding='utf-8') as fh:
 fh.write(html)

 if open_in_browser:
 try:
 webbrowser.open('file://' + str(html_path))
 except Exception:
 pass

 print(f'HTML salvo em: {html_path}')
 return str(html_path)

# rest unchanged
