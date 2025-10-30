# MIT License (c)2025 Andre Galberto - see LICENSE.md for full text
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os
import shutil
import subprocess
import sys
import time
import ast
import json
import plotly.express as px
import plotly.graph_objects as go # Para gráficos mais complexos

# --- Definições Globais e Funções Auxiliares ---

# Estrutura de Pastas (Relativas ao diretório do script)
script_dir = os.path.dirname(__file__)
INPUT_FOLDER = os.path.join(script_dir, 'input_genes')
OUTPUT_FOLDER = os.path.join(script_dir, 'simulation_outputs')
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Informações de Aminoácidos
AMINO_ACID_INFO = {
 'A': {'name': 'Alanina', 'prop': 'Não-polar, pequena'},
 'R': {'name': 'Arginina', 'prop': 'Carregado (+), polar'},
 'N': {'name': 'Asparagina', 'prop': 'Polar'},
 'D': {'name': 'Ácido Aspártico', 'prop': 'Carregado (-), polar'},
 'C': {'name': 'Cisteína', 'prop': 'Polar, especial (ponte S-S)'},
 'Q': {'name': 'Glutamina', 'prop': 'Polar'},
 'E': {'name': 'Ácido Glutâmico', 'prop': 'Carregado (-), polar'},
 'G': {'name': 'Glicina', 'prop': 'Não-polar, flexível'},
 'H': {'name': 'Histidina', 'prop': 'Carregado (+/-), polar'},
 'I': {'name': 'Isoleucina', 'prop': 'Não-polar, hidrofóbico'},
 'L': {'name': 'Leucina', 'prop': 'Não-polar, hidrofóbico'},
 'K': {'name': 'Lisina', 'prop': 'Carregado (+), polar'},
 'M': {'name': 'Metionina', 'prop': 'Não-polar'},
 'F': {'name': 'Fenilalanina', 'prop': 'Não-polar, aromático'},
 'P': {'name': 'Prolina', 'prop': 'Não-polar, especial (rígido)'},
 'S': {'name': 'Serina', 'prop': 'Polar'},
 'T': {'name': 'Treonina', 'prop': 'Polar'},
 'W': {'name': 'Triptofano', 'prop': 'Não-polar, aromático'},
 'Y': {'name': 'Tirosina', 'prop': 'Polar, aromático'},
 'V': {'name': 'Valina', 'prop': 'Não-polar, hidrofóbico'},
 'X': {'name': 'Desconhecido/Gap', 'prop': 'Indefinido'},
 '-': {'name': 'Gap', 'prop': 'Indefinido'},
 '*': {'name': 'Stop Codon', 'prop': 'Fim'},
}

def get_aa_info(aa_code):
 """Retorna nome e propriedade do aminoácido."""
 info = AMINO_ACID_INFO.get(aa_code.upper(), {'name': 'Inválido', 'prop': 'N/A'})
 return f"{info['name']} ({info['prop']})"

def find_mutation_details(original_prot, mutated_prot):
 """Encontra a primeira mudança entre duas sequências e retorna detalhes."""
 if not isinstance(original_prot, str) or not isinstance(mutated_prot, str):
 return "N/A"
 
 # Lidar com comprimentos diferentes (ex: stop codon prematuro)
 min_len = min(len(original_prot), len(mutated_prot))
 
 for i in range(min_len):
 if original_prot[i] != mutated_prot[i]:
 orig_info = AMINO_ACID_INFO.get(original_prot[i], {'name': original_prot[i]})
 mut_info = AMINO_ACID_INFO.get(mutated_prot[i], {'name': mutated_prot[i]})
 return f"Pos {i+1}: {orig_info['name']} ({original_prot[i]}) -> {mut_info['name']} ({mutated_prot[i]})"
 
 # Se nenhuma mudança encontrada nos primeiros min_len caracteres
 if len(original_prot) != len(mutated_prot):
 if len(mutated_prot) < len(original_prot):
 stop_info = AMINO_ACID_INFO.get(original_prot[min_len], {'name': original_prot[min_len]})
 return f"Stop Prematuro? Esperado {stop_info['name']} ({original_prot[min_len]}) na pos {min_len+1}"
 else:
 return f"Extensão? Seq. mutada mais longa."

 return "Sequências idênticas" # Nenhuma mudança encontrada

# --- Importações e Fallbacks (mantidos do código anterior) ---
try:
 from PythonIA import psi, f_ia, rodar_simulacao, a, c, m
except Exception:
 st.error("Erro ao importar PythonIA.py. Verifique a localização.")
 psi, f_ia, rodar_simulacao = None, None, None
 a, c, m =1664525,1013904223,2**32

sim_script_path = os.path.join(script_dir, 'simular_anticorpo.py')
if not os.path.exists(sim_script_path):
 project_root = os.path.dirname(script_dir)
 sim_script_path = os.path.join(project_root, 'simular_anticorpo.py')
 if not os.path.exists(sim_script_path):
 sim_script_path = None
 st.warning("Script 'simular_anticorpo.py' não encontrado.")

# --- Estado da Sessão (inicialização robusta) ---
def _init_session():
 default_session_state = {
 'sim_proc': None, 'sim_out': None, 'sim_err': None, 'sim_out_json': None,
 'sim_start_time': None, 'sim_result': None, 'sim_status': 'idle',
 'volterra_results': None # Para guardar resultados da simulação Volterra
 }
 for key, default_value in default_session_state.items():
 if key not in st.session_state:
 st.session_state[key] = default_value

_init_session()

# --- Layout do Dashboard ---
st.set_page_config(layout="wide") # Usa a largura total da página
st.title("🧬 Dashboard: Simulação & Engenharia de Proteínas")
st.markdown("---")

# --- Barra Lateral: Controles ---
with st.sidebar:
 st.header("⚙️ Configurações Gerais")

 # --- Controles Simulação Volterra ---
 with st.expander("Simulação Volterra (Paciente)", expanded=False):
 lambda_p = st.slider("Taxa de Decaimento (λ)",0.001,0.1,0.0154,0.001, key='lambda_volterra')
 xi_p = st.number_input("Semente Numérica (xi_p)", value=123456789, key='xi_volterra')
 n_dias = st.slider("Dias de Simulação",100,1000,365, key='dias_volterra')
 run_volterra = st.button("Executar Simulação Volterra")

 # --- Controles Engenharia de Anticorpo ---
 with st.expander("Engenharia de Anticorpo (IA)", expanded=True):
 model_path = st.text_input('Modelo Keras (.keras)', value='modelo_anticorpo_corretivo.keras', key='model_path_ia')
 n_mutacoes = st.number_input('Iterações (mutações)', min_value=1, max_value=50000, value=500, step=100, key='n_mutacoes_ia')

 st.subheader("Gene/Anticorpo Inicial")
 fasta_option = st.radio("Fonte do Gene:", ('Upload', f'Pasta "{os.path.basename(INPUT_FOLDER)}"', 'Nenhum (usar default)'), key='fasta_source', horizontal=True)

 fasta_filename_to_use = None
 if fasta_option == 'Upload':
 fasta_file = st.file_uploader("Arquivo FASTA", type=['fasta', 'fa'], key='fasta_upload')
 if fasta_file is not None:
 tmpdir = tempfile.gettempdir()
 fasta_path_temp = os.path.join(tmpdir, fasta_file.name)
 with open(fasta_path_temp, 'wb') as out_f: out_f.write(fasta_file.getbuffer())
 fasta_filename_to_use = fasta_path_temp
 st.success(f'Usando: {fasta_file.name}')
 elif fasta_option == f'Pasta "{os.path.basename(INPUT_FOLDER)}"':
 try:
 fasta_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.fasta', '.fa', '.fna'))])
 if fasta_files:
 selected_fasta = st.selectbox("Selecione o arquivo:", options=fasta_files, key='fasta_select')
 fasta_filename_to_use = os.path.join(INPUT_FOLDER, selected_fasta)
 st.info(f'Usando: {selected_fasta}')
 else:
 st.warning(f'Nenhum arquivo FASTA encontrado em "{INPUT_FOLDER}".')
 except FileNotFoundError:
 st.error(f'Pasta de entrada não encontrada: "{INPUT_FOLDER}"')
 else:
 st.info("Nenhum arquivo selecionado (usará default do script).")

 run_antibody_sim = st.button("Iniciar Engenharia de Anticorpo")

# --- Área Principal ---

# --- Execução e Exibição: Simulação Volterra ---
if run_volterra:
 if rodar_simulacao:
 with st.spinner("Executando Simulação Volterra..."):
 t, x, f, m_hist = rodar_simulacao(n_dias,1, lambda_p, xi_p, a, c, m) # dt=1
 st.session_state['volterra_results'] = pd.DataFrame({'Tempo (dias)': t, 'Estado (x_n)': x, 'Impacto (f_n)': f})
 st.experimental_rerun() # Recarrega para mostrar os resultados
 else:
 st.error("Função 'rodar_simulacao' não disponível.")

if st.session_state['volterra_results'] is not None:
 st.header(f"📈 Resultados Volterra (Semente {st.session_state.get('xi_p_last_run', xi_p)})") # Mostra a semente usada
 df_volterra = st.session_state['volterra_results']

 # Criar figura Plotly com dois eixos Y
 fig_volterra = go.Figure()
 # Linha para Estado (x_n) no eixo Y esquerdo
 fig_volterra.add_trace(go.Scatter(x=df_volterra['Tempo (dias)'], y=df_volterra['Estado (x_n)'],
 mode='lines', name='Estado (x_n)', yaxis='y1',
 line=dict(color=_color_estado)))
 # Barras para Impacto (f_n) no eixo Y direito
 fig_volterra.add_trace(go.Bar(x=df_volterra['Tempo (dias)'], y=df_volterra['Impacto (f_n)'],
 name='Impacto (f_n)', yaxis='y2',
 marker=dict(color=_color_impacto, opacity=0.6)))

 # Configurar layout com dois eixos Y
 fig_volterra.update_layout(
 title="Evolução Temporal: Estado vs. Impacto",
 xaxis_title="Tempo (dias)",
 yaxis=dict(
 title="Estado (x_n)",
 titlefont=dict(color="royalblue"),
 tickfont=dict(color="royalblue")
 ),
 yaxis2=dict(
 title="Impacto (f_n)",
 titlefont=dict(color="crimson"),
 tickfont=dict(color="crimson"),
 overlaying="y",
 side="right",
 showgrid=False # Não mostrar grid para o segundo eixo
 ),
 legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
 hovermode="x unified" # Mostra info de ambas as traces ao passar o mouse
 )
 st.plotly_chart(fig_volterra, use_container_width=True)

 with st.expander("Ver Tabela de Dados Volterra (últimos10 dias)"):
 st.dataframe(df_volterra.tail(10))
 st.markdown("---")


# --- Execução e Exibição: Engenharia de Anticorpo ---
if run_antibody_sim:
 if st.session_state['sim_status'] == 'running':
 st.warning("Simulação de anticorpo já em execução.")
 elif sim_script_path is None:
 st.error("Script 'simular_anticorpo.py' não encontrado.")
 else:
 uid = str(int(time.time()))
 out_log = os.path.join(OUTPUT_FOLDER, f'sim_ant_out_{uid}.log')
 err_log = os.path.join(OUTPUT_FOLDER, f'sim_ant_err_{uid}.log')
 out_json = os.path.join(OUTPUT_FOLDER, f'sim_ant_result_{uid}.json')

 cmd = [sys.executable, sim_script_path,
 '--model', model_path,
 '--n', str(int(n_mutacoes)),
 '--out-json', out_json]
 if fasta_filename_to_use:
 cmd += ['--dna-file', fasta_filename_to_use]

 try:
 with open(out_log, 'w', encoding='utf-8') as out_f, \
 open(err_log, 'w', encoding='utf-8') as err_f:
 proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f,
 cwd=os.path.dirname(sim_script_path),
 universal_newlines=True)
 st.session_state['sim_proc'] = proc
 st.session_state['sim_out'] = out_log
 st.session_state['sim_err'] = err_log
 st.session_state['sim_out_json'] = out_json
 st.session_state['sim_start_time'] = time.time()
 st.session_state['sim_status'] = 'running'
 st.session_state['sim_result'] = None
 st.success(f'Simulação de anticorpo iniciada (PID {proc.pid}). Logs em: {OUTPUT_FOLDER}')
 st.experimental_rerun()
 except Exception as e:
 st.error(f'Falha ao iniciar simulação de anticorpo: {e}')

# Monitoramento e exibição de resultados do subprocesso (IA)
st.header("🔬 Resultados da Engenharia de Anticorpo")
status_placeholder_ia = st.empty()
log_expander = st.expander("Ver Logs da Simulação", expanded=False)

if st.session_state['sim_status'] == 'running':
 proc = st.session_state.get('sim_proc')
 if proc:
 poll_result = proc.poll()
 if poll_result is None:
 elapsed = time.time() - st.session_state['sim_start_time']
 status_placeholder_ia.info(f'Executando... Tempo: {elapsed:.0f}s. (Clique em "Ver Status" na barra lateral para atualizar)')
 # Mostrar log dentro do expander
 try:
 if st.session_state['sim_out'] and os.path.exists(st.session_state['sim_out']):
 with open(st.session_state['sim_out'], 'r', encoding='utf-8', errors='ignore') as f:
 lines = f.readlines()
 with log_expander:
 st.text_area("Log (stdout):", "".join(lines[-20:]), height=250, key='log_area_run')
 except Exception as e:
 with log_expander: st.warning(f"Não foi possível ler o log: {e}")
 # Rerun periódico pode ser irritante, remover ou espaçar mais
 # time.sleep(5)
 # st.experimental_rerun()
 else: # Processo terminou
 st.session_state['sim_status'] = 'finished'
 st.session_state['sim_proc'] = None
 status_placeholder_ia.success(f'Simulação finalizada (código: {poll_result}). Carregando resultados...')
 st.experimental_rerun() # Força rerun para carregar e mostrar resultados

elif st.session_state['sim_status'] in ('finished', 'cancelled', 'error'):
 if st.session_state['sim_status'] == 'cancelled': status_placeholder_ia.warning("Simulação foi cancelada.")
 elif st.session_state['sim_status'] == 'error': status_placeholder_ia.error("Erro durante a simulação.")

 out_json_path = st.session_state.get('sim_out_json')

 # Tenta carregar o resultado do JSON se ainda não estiver na sessão
 if not st.session_state.get('sim_result') and out_json_path and os.path.exists(out_json_path):
 try:
 with open(out_json_path, 'r', encoding='utf-8') as jf:
 st.session_state['sim_result'] = json.load(jf)
 status_placeholder_ia.success(f"Resultados carregados de: {os.path.basename(out_json_path)}")
 except Exception as e:
 status_placeholder_ia.error(f'Falha ao ler JSON ({os.path.basename(out_json_path)}): {e}')
 st.session_state['sim_result'] = None

 # Exibe os resultados se disponíveis
 if st.session_state['sim_result']:
 parsed = st.session_state['sim_result']
 
 # Mostrar Resumo
 st.subheader("📊 Resumo da Otimização")
 col1_res, col2_res = st.columns(2)
 with col1_res:
 st.metric("Impacto Original", f"{parsed.get('original_impact',0):.4f}")
 st.markdown(f"**Proteína Original:**")
 st.code(parsed.get('original_protein', 'N/A'), language=None)
 with col2_res:
 st.metric("Melhor Impacto Encontrado", f"{parsed.get('best_impact',0):.4f}")
 st.markdown(f"**Melhor Proteína:**")
 st.code(parsed.get('best_protein', 'N/A'), language=None)
 
 # Detalhes da Melhor Mutação
 best_mut_details = find_mutation_details(parsed.get('original_protein'), parsed.get('best_protein'))
 st.markdown(f"**Melhor Mutação Identificada:** {best_mut_details}")


 # Análise Interativa do Histórico
 st.subheader("📈 Análise Interativa do Histórico")
 history = parsed.get('history', [])
 if history:
 df_hist = pd.DataFrame(history)
 try:
 if not df_hist.empty:
 df_hist['impacto'] = pd.to_numeric(df_hist['impacto'], errors='coerce').fillna(0)
 df_hist['i'] = pd.to_numeric(df_hist['i'], errors='coerce').fillna(0)
 df_hist['prot_len'] = df_hist['prot'].apply(lambda x: len(x) if isinstance(x, str) else0)

 #1. Gráfico de Impacto vs Iteração com Melhor Cumulativo
 df_hist['melhor_impacto_acumulado'] = df_hist['impacto'].cummax()
 fig_progress = go.Figure()
 fig_progress.add_trace(go.Scatter(x=df_hist['i'], y=df_hist['impacto'], mode='markers', name='Impacto da Iteração',
 marker=dict(color='lightblue', size=5, opacity=0.7),
 customdata=df_hist[['prot', 'dna']],
 hovertemplate="Iter: %{x}<br>Impacto: %{y:.4f}<br>Prot: %{customdata[0]}<extra></extra>"))
 fig_progress.add_trace(go.Scatter(x=df_hist['i'], y=df_hist['melhor_impacto_acumulado'], mode='lines', name='Melhor Impacto Acumulado',
 line=dict(color=_color_progress, width=2)))
 fig_progress.update_layout(title="Progresso da Otimização: Impacto vs Iteração",
 xaxis_title="Iteração (i)", yaxis_title="Impacto Previsto (f)",
 hovermode="closest")
 st.plotly_chart(fig_progress, use_container_width=True)

 #2. Histograma e Top N (com filtro)
 min_imp_hist = float(df_hist['impacto'].min())
 max_imp_hist = float(df_hist['impacto'].max())
 default_thresh_hist = max(min_imp_hist, max_imp_hist *0.1) if max_imp_hist > min_imp_hist else min_imp_hist
 threshold_hist = st.slider('Filtrar Histograma/Tabela por Impacto Mínimo', min_imp_hist, max_imp_hist, default_thresh_hist, key='hist_slider')

 df_filtered_hist = df_hist[df_hist['impacto'] >= threshold_hist].copy()

 if not df_filtered_hist.empty:
 # Adiciona detalhes da mutação e aminoácidos
 orig_prot_hist = parsed.get('original_protein')
 df_filtered_hist['mutacao_detalhe'] = df_filtered_hist.apply(lambda row: find_mutation_details(orig_prot_hist, row['prot']), axis=1)

 col_hist, col_top = st.columns(2)
 with col_hist:
 fig_hist_dist = px.histogram(df_filtered_hist, x='impacto', nbins=30, title='Distribuição dos Impactos Filtrados', color_discrete_sequence=[_color_progress])
 st.plotly_chart(fig_hist_dist, use_container_width=True)
 
 with col_top:
 top_n_hist = st.number_input('Mostrar top N mutações (filtradas)', min_value=1, max_value=len(df_filtered_hist), value=min(10, len(df_filtered_hist)), key='hist_topn')
 top_df_hist = df_filtered_hist.sort_values('impacto', ascending=False).head(int(top_n_hist))
 st.markdown(f'**Top {int(top_n_hist)} Mutações (Impacto >= {threshold_hist:.3f})**')
 # Selecionar e renomear colunas para a tabela
 st.dataframe(top_df_hist[['i', 'prot', 'impacto', 'mutacao_detalhe']].rename(columns={'i': 'Iter', 'prot': 'Proteína', 'impacto': 'Impacto', 'mutacao_detalhe': 'Mutação'}).reset_index(drop=True))

 # Visualização do aminoácido chave (primeira diferença) para o top hit
 try:
 if not top_df_hist.empty:
 top_hit = top_df_hist.iloc[0]
 orig_prot = parsed.get('original_protein', '')
 mut_prot = top_hit['prot']
 # find first differing position
 pos = None
 for idx in range(min(len(orig_prot), len(mut_prot))):
 if orig_prot[idx] != mut_prot[idx]:
 pos = idx +1
 break
 if pos:
 aa_orig = orig_prot[pos -1]
 aa_mut = mut_prot[pos -1]
 col_a, col_b = st.columns(2)
 with col_a:
 st.markdown(f"**Aminoácido Original (pos {pos})**: {aa_orig}")
 st.write(get_aa_info(aa_orig))
 with col_b:
 st.markdown(f"**Aminoácido Mutado (pos {pos})**: {aa_mut}")
 st.write(get_aa_info(aa_mut))
 # simple interpretation
 interp = []
 if 'Carregado' in AMINO_ACID_INFO.get(aa_orig, {}).get('prop', '') and 'Carregado' not in AMINO_ACID_INFO.get(aa_mut, {}).get('prop', ''):
 interp.append('Perda de carga pode reduzir interações eletrostáticas')
 if 'hidrofóbico' in AMINO_ACID_INFO.get(aa_mut, {}).get('prop', '').lower():
 interp.append('Aumento de hidrofobicidade pode afetar dobra local')
 if interp:
 st.info(' | '.join(interp))
 except Exception:
 pass

# Add configuration persistence and UI color settings
CONFIG_PATH = os.path.join(script_dir, 'dashboard_config.json')

def load_config():
 default = {
 'model_path': 'modelo_anticorpo_corretivo.keras',
 'n_mutacoes':500,
 'color_estado': '#4169E1', # royalblue
 'color_impacto': '#DC143C', # crimson
 'color_progress': '#FF4136'
 }
 try:
 if os.path.exists(CONFIG_PATH):
 with open(CONFIG_PATH, 'r', encoding='utf-8') as cf:
 data = json.load(cf)
 default.update(data)
 except Exception:
 pass
 return default

def save_config(cfg):
 try:
 with open(CONFIG_PATH, 'w', encoding='utf-8') as cf:
 json.dump(cfg, cf, ensure_ascii=False, indent=2)
 return True
 except Exception:
 return False

config = load_config()

# Sidebar additions for styles and preference saving
with st.sidebar.expander('Aparência e Preferências', expanded=False):
 st.subheader('Cores dos Gráficos')
 color_estado = st.color_picker('Cor do Estado (x_n)', value=config.get('color_estado', '#4169E1'))
 color_impacto = st.color_picker('Cor do Impacto (f_n)', value=config.get('color_impacto', '#DC143C'))
 color_progress = st.color_picker('Cor do Progresso', value=config.get('color_progress', '#FF4136'))
 st.write('Preferências do Modelo')
 pref_model = st.text_input('Caminho padrão do modelo', value=config.get('model_path', 'modelo_anticorpo_corretivo.keras'))
 pref_nmut = st.number_input('Iterações padrão', value=int(config.get('n_mutacoes',500)), min_value=1, max_value=100000, step=10)
 if st.button('Salvar Preferências'):
 cfg = {
 'model_path': pref_model,
 'n_mutacoes': pref_nmut,
 'color_estado': color_estado,
 'color_impacto': color_impacto,
 'color_progress': color_progress
 }
 ok = save_config(cfg)
 if ok:
 st.success('Preferências salvas')
 else:
 st.error('Falha ao salvar preferências')

# Use selected colors where figures are created later; set defaults from config if not overridden
_color_estado = color_estado if 'color_estado' in locals() else config.get('color_estado', '#4169E1')
_color_impacto = color_impacto if 'color_impacto' in locals() else config.get('color_impacto', '#DC143C')
_color_progress = color_progress if 'color_progress' in locals() else config.get('color_progress', '#FF4136')

# --- Helpers: list, delete, thumbnail generation for past runs ---
def list_result_jsons(outdir=OUTPUT_FOLDER):
 files = []
 try:
 for fn in sorted(os.listdir(outdir), reverse=True):
 if fn.endswith('.json') and fn.startswith('sim_ant_result_'):
 path = os.path.join(outdir, fn)
 try:
 mtime = os.path.getmtime(path)
 except Exception:
 mtime =0
 files.append({'name': fn, 'path': path, 'mtime': mtime})
 except Exception:
 pass
 files.sort(key=lambda x: x['mtime'], reverse=True)
 return files


def uid_from_result_name(name):
 # expected name like sim_ant_result_<uid>.json
 base = os.path.basename(name)
 if base.startswith('sim_ant_result_') and base.endswith('.json'):
 return base[len('sim_ant_result_'):-len('.json')]
 return None


def generate_thumbnail_for_result(json_path, outdir=OUTPUT_FOLDER):
 """Generate a small PNG thumbnail summarizing the result (histogram of impacts).
 Returns thumbnail path or None.
 """
 try:
 with open(json_path, 'r', encoding='utf-8') as jf:
 data = json.load(jf)
 history = data.get('history', [])
 if not history:
 return None
 impacts = [float(h.get('impacto', h.get('impact',0))) for h in history]
 if not impacts:
 return None
 # create thumbnails dir
 thumbs_dir = os.path.join(outdir, 'thumbnails')
 os.makedirs(thumbs_dir, exist_ok=True)
 base = os.path.basename(json_path).replace('.json', '.png')
 thumb_path = os.path.join(thumbs_dir, f"thumb_{base}")
 # small matplotlib figure
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig = plt.figure(figsize=(2.5,1.5), dpi=100)
 ax = fig.add_subplot(111)
 ax.hist(impacts, bins=10, color='#4C72B0')
 ax.set_xticks([])
 ax.set_yticks([])
 ax.set_title('Impacts', fontsize=8)
 fig.tight_layout()
 fig.savefig(thumb_path, bbox_inches='tight')
 plt.close(fig)
 return thumb_path
 except Exception:
 return None


def delete_run_and_artifacts(json_info):
 """Delete JSON result and associated logs and thumbnail.
 json_info is dict with 'name' and 'path'."""
 try:
 path = json_info['path']
 uid = uid_from_result_name(json_info['name'])
 # delete json
 try:
 os.remove(path)
 except Exception:
 pass
 # delete logs
 out_log = os.path.join(OUTPUT_FOLDER, f'sim_ant_out_{uid}.log')
 err_log = os.path.join(OUTPUT_FOLDER, f'sim_ant_err_{uid}.log')
 for p in (out_log, err_log):
 try:
 if os.path.exists(p):
 os.remove(p)
 except Exception:
 pass
 # delete thumbnail
 thumb = os.path.join(OUTPUT_FOLDER, 'thumbnails', f"thumb_{json_info['name'].replace('.json','.png')}")
 try:
 if os.path.exists(thumb):
 os.remove(thumb)
 except Exception:
 pass
 return True
 except Exception:
 return False

# --- Sidebar: previous runs selector and management ---
available_runs = list_result_jsons()
if available_runs:
 run_options = [f"{r['name']} ({time.ctime(r['mtime'])})" for r in available_runs]
 selected_idx = st.sidebar.selectbox('Selecionar execução anterior', options=list(range(len(run_options))), format_func=lambda i: run_options[i], index=0, key='select_prev_run')
 selected_run = available_runs[selected_idx]
 if st.sidebar.button('Carregar execução selecionada'):
 try:
 with open(selected_run['path'], 'r', encoding='utf-8') as jf:
 st.session_state['sim_result'] = json.load(jf)
 st.session_state['sim_out_json'] = selected_run['path']
 st.session_state['sim_status'] = 'finished'
 st.success(f'Execução {selected_run["name"]} carregada')
 st.experimental_rerun()
 except Exception as e:
 st.error(f'Falha ao carregar execução: {e}')

 # delete controls
 confirm_del = st.sidebar.checkbox('Confirmar remoção da execução selecionada', value=False, key='confirm_del')
 if st.sidebar.button('Deletar execução selecionada'):
 if not confirm_del:
 st.sidebar.warning('Marque a confirmação antes de deletar.')
 else:
 ok = delete_run_and_artifacts(selected_run)
 if ok:
 st.sidebar.success('Execução e artefatos deletados.')
 st.experimental_rerun()
 else:
 st.sidebar.error('Falha ao deletar a execução selecionada.')

 # thumbnail generation
 if st.sidebar.button('Gerar miniatura para execução selecionada'):
 thumb = generate_thumbnail_for_result(selected_run['path'])
 if thumb:
 st.sidebar.success(f'Miniatura criada: {os.path.relpath(thumb)}')
 else:
 st.sidebar.warning('Não foi possível criar miniatura para esta execução.')

 if st.sidebar.button('Gerar miniaturas para todas as execuções'):
 created =0
 for r in available_runs:
 if generate_thumbnail_for_result(r['path']):
 created +=1
 st.sidebar.success(f'{created} miniaturas criadas.')

else:
 st.sidebar.info('Nenhuma execução anterior encontrada em OUTPUT_FOLDER.')

# continue with rest of UI...