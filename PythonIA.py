# MIT License (c)2025 Andre Galberto - see LICENSE.md for full text
# -*- coding: cp1252 -*-
import sys
# Ensure stdout uses UTF-8 to avoid garbled non-ASCII output in many consoles
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import os
import numpy as np
import matplotlib
# Use Agg backend to ensure headless environments can save figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 1. PARAMETROS DO MODELO (Baseado no seu artigo) ---

# Default parameters
lambda_p = 0.01540  # default decay (day^-1)
dt = 1             # discretization (1 day)
n_dias = 365       # total days

# Biological seed example
xi_p = 123456789

# GLC parameters (ANSI C-like)
a = 1103515245
c = 12345
m = 2**31 - 1

# --- helper functions ---

def compute_lambda_for_tumor(tumor_type: str, stage: int, alpha: float = 0.0):
    """Return lambda_p for given tumor type and (integer) stage.
    If alpha>0, use formula lambda_p = lambda0 / (1 + alpha*S_p) where S_p is severity in [0,1].
    Otherwise use suggested table values for common tumor types.
    """
    # table from the article (approx)
    table = {
        'prostata': { (1,2): 0.00385, (3,4): 0.00770 },
        'mama':     { (1,2): 0.00580 },
        'pancreas': { (3,4): 0.01540 },
    }
    key = tumor_type.lower()
    if alpha and 0 <= stage <= 4:
        # simple severity mapping: Sp = (stage-1)/3 for stage in 1..4
        Sp = max(0.0, min(1.0, (stage-1)/3.0))
        # base lambda0 choose pancreas default if unknown
        lambda0 = 0.01540
        return lambda0 / (1.0 + alpha * Sp)

    if key in table:
        mapping = table[key]
        # choose appropriate bucket
        for stages, val in mapping.items():
            if stage in range(stages[0], stages[-1]+1 if isinstance(stages, tuple) else stages+1):
                return val
        # fallback to first value
        return list(mapping.values())[0]
    # default fallback
    return 0.01540


def seed_to_glc_x0(xi_p):
    """Convert biological seed xi_p (int or str) to initial GLC state X0.
    If xi_p is numeric, use it; if string, hash.
    """
    try:
        # if numeric
        X0 = int(xi_p) & 0x7fffffff
    except Exception:
        X0 = abs(hash(str(xi_p))) % m
    return X0


# --- 2. FUNCOES PRINCIPAIS (IA e ESTADO INICIAL) ---

def psi(semente_paciente):
    # simple initial state
    return (int(semente_paciente) % 100) / 100.0


def f_ia(memoria_passada, tempo, semente_paciente, estado_aleatorio):
    # Normalize random state
    ruido_mutacao = estado_aleatorio / m
    impacto_base = 0.05
    if ruido_mutacao > 0.95:
        return impacto_base + ruido_mutacao * 5
    return impacto_base + ruido_mutacao


# --- 3. SIMULATION ---

def rodar_simulacao(n_dias, dt, lambda_p, xi_p, a, c, m):
    """Runs the recursive simulation and returns times, x_hist, f_hist, m_hist, X_hist"""
    tempos = np.arange(0, n_dias, dt)
    n_passos = len(tempos)
    X_hist = np.zeros(n_passos, dtype=np.int64)
    f_hist = np.zeros(n_passos)
    m_hist = np.zeros(n_passos)
    x_hist = np.zeros(n_passos)

    x_0 = psi(xi_p)
    x_hist[0] = x_0
    m_hist[0] = 0
    X_hist[0] = seed_to_glc_x0(xi_p)
    f_hist[0] = f_ia(0, 0, xi_p, X_hist[0])

    decaimento = np.exp(-lambda_p * dt)

    for n in range(1, n_passos):
        X_n = (a * int(X_hist[n-1]) + c) % m
        f_n = f_ia(x_hist[:n], tempos[n], xi_p, X_n)
        m_n = decaimento * m_hist[n-1] + f_n * dt
        x_n = x_0 + m_n
        X_hist[n] = X_n
        f_hist[n] = f_n
        m_hist[n] = m_n
        x_hist[n] = x_n

    return tempos, x_hist, f_hist, m_hist, X_hist


def run_simulation_with_detection(n_dias, dt, lambda_p, xi_p, a, c, m, threshold=0.5):
    """Run simulation and detect mutation events where f_n > threshold.
    Returns (tempos, x_hist, f_hist, m_hist, X_hist, events)
    events is a list of dicts: {'step': n, 'time': t, 'f': f_n, 'X': X_n}
    """
    tempos, x_hist, f_hist, m_hist, X_hist = rodar_simulacao(n_dias, dt, lambda_p, xi_p, a, c, m)
    events = []
    for n, f_n in enumerate(f_hist):
        if f_n > threshold:
            events.append({'step': int(n), 'time': float(tempos[n]), 'f': float(f_n), 'X': int(X_hist[n])})
    return tempos, x_hist, f_hist, m_hist, X_hist, events


def gerar_grafico(tempos, x_hist, f_hist):
    """
    Gera o "Dashboard" estatico. Valida entradas, salva com caminho absoluto e fecha figura.
    """
    print("Gerando grafico...")

    # Valida entradas
    if tempos is None or x_hist is None or f_hist is None:
        print("Erro: argumentos None recebidos em gerar_grafico.")
        return
    if len(tempos) == 0 or len(x_hist) == 0 or len(f_hist) == 0:
        print(f"Aviso: arrays vazios. len(tempos)={len(tempos)}, len(x_hist)={len(x_hist)}, len(f_hist)={len(f_hist)}")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Titulo do Dashboard
    fig.suptitle("Simulacao do Modelo Volterra-Stieltjes com Memoria Finita", fontsize=16)

    # --- Grafico 1: Estado do Paciente (x_n) ---
    color = 'tab:blue'
    ax1.set_xlabel('Tempo (dias)', fontsize=12)
    ax1.set_ylabel('Estado do Paciente (x_n)', color=color, fontsize=12)
    ax1.plot(tempos, x_hist, color=color, label="Estado x_n (Evolucao)")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Grafico 2: Impacto da IA (f_n) ---
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Impacto da IA (f_n)', color=color, fontsize=12)
    ax2.bar(tempos, f_hist, color=color, alpha=0.3, label="Impacto f_n (Mutacoes)")
    ax2.tick_params(axis='y', labelcolor=color)

    # Salvar o grafico em caminho absoluto e fechar figura
    nome_arquivo = os.path.join(os.getcwd(), "dashboard_simulacao_paciente.png")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(nome_arquivo)
    plt.close(fig)

    # Tentar abrir automaticamente no Windows
    try:
        if os.name == 'nt':
            os.startfile(nome_arquivo)
    except Exception as e:
        print("Não foi possível abrir automaticamente:", e)

    # Confirmação
    if os.path.exists(nome_arquivo):
        try:
            tamanho = os.path.getsize(nome_arquivo)
        except Exception:
            tamanho = -1
        print(f"Dashboard salvo em: {nome_arquivo} (tamanho: {tamanho} bytes)")
    else:
        print(f"Erro ao salvar dashboard: arquivo não encontrado em {nome_arquivo}")


# --- EXECUTAR O PROGRAMA ---
if __name__ == "__main__":
    # Desempacotar todos os retornos e evitar sobrescrever o parametro 'm'
    t, x, f, m_hist, X_hist = rodar_simulacao(n_dias, dt, lambda_p, xi_p, a, c, m)
    # imprimir comprimentos para debug rapido
    print(f"len: t={len(t)}, x={len(x)}, f={len(f)}")
    gerar_grafico(t, x, f)