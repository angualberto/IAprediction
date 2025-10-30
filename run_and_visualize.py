# MIT License (c)2025 Andre Galberto - see LICENSE.md for full text
"""
Run simulation (Volterra-Stieltjes discretized) with event detection and attempt to visualize detected mutation
using the p53 visualizer. This version maps detected stochastic states Xn to residue numbers deterministically
and uses a lower default detection threshold.

Usage (CLI):
  python PythonIA/run_and_visualize.py --days 365 --threshold 0.2

Mapping strategy:
  residue = (Xn % protein_length) + 1

If py3Dmol is not available or the environment is not a Jupyter notebook, the script will save the PDB under
PythonIA/pdbs/ and print instructions for opening the visualization in a notebook.
"""

import argparse
import os
import sys
from pprint import pprint
import webbrowser

# import simulation functions
try:
    from PythonIA import run_simulation_with_detection
except Exception:
    # try relative import
    import importlib.util
    spec = importlib.util.spec_from_file_location('PythonIA', os.path.join(os.path.dirname(__file__), 'PythonIA.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_simulation_with_detection = mod.run_simulation_with_detection


def try_visualize_event(pdb_id='2OCJ', chain='A', resi=248, mutation='R248W'):
    """Try to import visualize_p53_r248w and display or save PDB.
    Returns a dict with keys: 'view_available' (bool), 'saved_pdb' (path or None), 'html' (path or None).
    """
    result = {'view_available': False, 'saved_pdb': None, 'html': None}
    try:
        from PythonIA.visualize_p53_r248w import visualize_mutation, export_html_view
    except Exception:
        try:
            # try load by path
            import importlib.util
            viz_path = os.path.join(os.path.dirname(__file__), 'visualize_p53_r248w.py')
            spec = importlib.util.spec_from_file_location('visualize_p53_r248w', viz_path)
            vizmod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vizmod)
            visualize_mutation = vizmod.visualize_mutation
            export_html_view = getattr(vizmod, 'export_html_view', None)
        except Exception:
            print('visualize_p53_r248w module not found or failed to import. Skipping live visualization.')
            return result

    # We have visualize_mutation function. Attempt to run it with download+save so PDB is persistent.
    try:
        view = visualize_mutation(pdb_id=pdb_id, chain=chain, resi=resi, mutation=mutation,
                                  download_pdb=True, save_pdb_in_project=True, label_lang='en')
        # In notebook, view.show() will render. Here we cannot guarantee notebook.
        # Only mark view_available if a live view object was actually returned.
        result['view_available'] = (view is not None)
        # The visualizer prints the saved path; attempt to compute expected saved path
        saved_path = os.path.join(os.path.dirname(__file__), 'pdbs', f'{pdb_id.lower()}.pdb')
        if os.path.isfile(saved_path):
            result['saved_pdb'] = saved_path
            # generate standalone HTML using export_html_view if available
            try:
                with open(saved_path, 'r', encoding='utf-8') as fh:
                    pdb_text = fh.read()
                if export_html_view is not None:
                    html_path = export_html_view(pdb_id, pdb_text, chain=chain, resi=resi, mutation=mutation)
                    result['html'] = html_path
                else:
                    # fallback: create a standalone HTML using 3Dmol CDN embedding
                    try:
                        import json
                        def _export_html_fallback(pdb_id, pdb_text, chain='A', resi=248, outdir=None):
                            if outdir is None:
                                outdir = os.path.join(os.path.dirname(__file__), 'pdbs')
                            os.makedirs(outdir, exist_ok=True)
                            safe_id = str(pdb_id).lower()
                            html_path = os.path.join(outdir, f"{safe_id}_{resi}.html")
                            pdb_js_string = json.dumps(pdb_text)
                            html = """<!doctype html>
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
 try {{
 viewer.addStyle({{chain: '{chain}', resi: {resi}}}, {{stick: {{colorscheme: 'magentaCarbon'}}, sphere: {{radius:1.0, color:'magenta'}}}});
 viewer.zoomTo({{chain: '{chain}', resi: {resi}}});
 }} catch(e) {{}}
 viewer.render();
 </script>
</body>
</html>""".format(pdb_js_string=pdb_js_string, safe_id=safe_id, chain=chain, resi=resi)
                            with open(html_path, 'w', encoding='utf-8') as fh:
                                fh.write(html)
                            try:
                                webbrowser.open('file://' + os.path.abspath(html_path))
                            except Exception:
                                pass
                            return html_path
                        # call fallback exporter
                        html_path = _export_html_fallback(pdb_id, pdb_text, chain=chain, resi=resi)
                        result['html'] = html_path
                    except Exception as e:
                        print('Fallback HTML generation failed:', e)
            except Exception as e:
                print('Failed to generate HTML view:', e)
    except Exception as e:
        print(f'Visualization call failed: {e}')
    return result


def map_X_to_residue(X_value, protein_length=393, offset=0):
    """Deterministic mapping from integer GLC state Xn to residue index in [1, protein_length].
    offset allows shifting the mapping reproducibly.
    """
    try:
        xv = int(X_value)
    except Exception:
        xv = abs(hash(str(X_value)))
    return (xv + int(offset)) % int(protein_length) + 1


def main(args):
    tempos, x_hist, f_hist, m_hist, X_hist, events = run_simulation_with_detection(
        n_dias=args.days, dt=1, lambda_p=args.lambda_p, xi_p=args.xi_p, a=args.a, c=args.c, m=args.m, threshold=args.threshold)

    print('\nSimulation finished. Detected events:')
    pprint(events)

    if not events:
        print('No mutation-like events detected above threshold.')
        return 0

    # Map events to residues
    for ev in events:
        ev['mapped_residue'] = map_X_to_residue(ev.get('X', 0), protein_length=args.protein_length, offset=args.mapping_offset)

    print('\nEvents mapped to residues:')
    pprint(events)

    # Choose first event for visualization
    ev = events[0]
    target_residue = int(ev.get('mapped_residue', args.residue))
    print(f"\nAttempting to visualize first detected event mapped to residue {target_residue}.")
    viz = try_visualize_event(pdb_id=args.pdb_id, chain=args.chain, resi=target_residue, mutation=args.mutation)

    if viz.get('view_available'):
        print('Visualizer available. If running in a Jupyter Notebook, import and call visualize_mutation or run this script there to see the 3D view.')
    else:
        print('Visualizer not available in this environment. PDB (if downloaded) saved at:')
        if viz.get('saved_pdb'):
            print('  ', viz['saved_pdb'])
        else:
            print('  No PDB saved. You can download PDB manually from RCSB: https://www.rcsb.org/')

    # Build a simple HTML dashboard combining the generated PNG (time series) and the 3D view HTML (if any)
    def export_dashboard_html(png_path, html3d_path=None, outdir=None, open_in_browser=True):
        if outdir is None:
            outdir = os.path.join(os.path.dirname(__file__), 'pdbs')
        os.makedirs(outdir, exist_ok=True)
        dashboard_path = os.path.join(outdir, f'dashboard_{args.pdb_id.lower()}_{target_residue}.html')

        # Resolve absolute file URLs for embedding
        png_url = None
        if png_path and os.path.exists(png_path):
            png_url = 'file://' + os.path.abspath(png_path)
        html3d_url = None
        if html3d_path and os.path.exists(html3d_path):
            html3d_url = 'file://' + os.path.abspath(html3d_path)

        parts = []
        parts.append('<!doctype html>')
        parts.append('<html><head><meta charset="utf-8"><title>Simulation Dashboard</title></head><body>')
        parts.append(f'<h1>Simulation Dashboard - PDB {args.pdb_id} res {target_residue}</h1>')
        if png_url:
            parts.append(f'<h2>Time series</h2><img src="{png_url}" style="max-width:100%;height:auto;border:1px solid #ccc;"/>')
        else:
            parts.append('<p><em>PNG time series not found.</em></p>')

        if html3d_url:
            parts.append('<h2>3D View</h2>')
            parts.append(f'<iframe src="{html3d_url}" style="width:100%;height:600px;border:1px solid #666;\"></iframe>')
        elif viz.get('saved_pdb'):
            parts.append('<p>PDB saved but interactive HTML not available. Open PDB with a viewer or use the exported HTML tool.</p>')
        else:
            parts.append('<p>No 3D view was generated.</p>')

        parts.append('</body></html>')

        with open(dashboard_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(parts))

        if open_in_browser:
            try:
                webbrowser.open('file://' + os.path.abspath(dashboard_path))
            except Exception:
                pass

        print('Dashboard HTML salvo em:', dashboard_path)
        return dashboard_path

    # Attempt to locate the PNG generated by the simulation
    png_candidates = [
        os.path.join(os.getcwd(), 'dashboard_simulacao_paciente.png'),
        os.path.join(os.path.dirname(__file__), 'dashboard_simulacao_paciente.png'),
    ]
    png_path = None
    for p in png_candidates:
        if os.path.exists(p):
            png_path = p
            break

    html3d = viz.get('html')
    # If html3d is None but saved_pdb exists and export_html_view is available, user will already have been provided with HTML
    if png_path or html3d:
        try:
            dashboard_file = export_dashboard_html(png_path, html3d)
        except Exception as e:
            print('Falha ao gerar dashboard HTML:', e)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulation and visualize detected mutation events (prototype)')
    parser.add_argument('--days', type=int, default=365, help='Number of days/steps to simulate')
    parser.add_argument('--threshold', type=float, default=0.2, help='Threshold on f_n to consider an event (lower default)')
    parser.add_argument('--lambda_p', type=float, default=0.0154, help='Decay lambda to use in simulation')
    parser.add_argument('--xi_p', default=123456789, help='Biological seed xi_p (int or str)')
    parser.add_argument('--a', type=int, default=1103515245, help='GLC parameter a')
    parser.add_argument('--c', type=int, default=12345, help='GLC parameter c')
    parser.add_argument('--m', type=int, default=(2**31 - 1), help='GLC parameter m')
    parser.add_argument('--pdb_id', type=str, default='2OCJ', help='PDB ID to visualize when events are found')
    parser.add_argument('--chain', type=str, default='A', help='Chain to highlight')
    parser.add_argument('--residue', type=int, default=248, help='Residue number to highlight (used if mapping disabled)')
    parser.add_argument('--mutation', type=str, default='R248W', help='Mutation label')
    parser.add_argument('--protein_length', type=int, default=393, help='Protein length for mapping Xn -> residue')
    parser.add_argument('--mapping_offset', type=int, default=0, help='Offset added to Xn before modulus to change mapping')

    args = parser.parse_args()
    sys.exit(main(args))
