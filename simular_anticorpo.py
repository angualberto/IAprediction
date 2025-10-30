# MIT License (c)2025 Andre Galberto - see LICENSE.md for full text
import os
import random
import hashlib
import numpy as np
import json
import tempfile

# Try to import TensorFlow/Keras; fall back to a dummy predictor if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Genetic code table (reuse)
CODON_TABLE = {
    'ATA':'I','ATC':'I','ATT':'I','ATG':'M','ACA':'T','ACC':'T','ACG':'T','ACT':'T',
    'AAC':'N','AAT':'N','AAA':'K','AAG':'K','AGC':'S','AGT':'S','TCA':'S','TCC':'S',
    'TCG':'S','TCT':'S','TTC':'F','TTT':'F','TTA':'L','TTG':'L','TAC':'Y','TAT':'Y',
    'TAA':'_','TAG':'_','TGC':'C','TGT':'C','TGA':'_','TGG':'W','CTA':'L','CTC':'L',
    'CTG':'L','CTT':'L','CCA':'P','CCC':'P','CCG':'P','CCT':'P','CAC':'H','CAT':'H',
    'CAA':'Q','CAG':'Q','CGA':'R','CGC':'R','CGG':'R','CGT':'R','GTA':'V','GTC':'V',
    'GTG':'V','GTT':'V','GCA':'A','GCC':'A','GCG':'A','GCT':'A','GAC':'D','GAT':'D',
    'GAA':'E','GAG':'E','GGA':'G','GGC':'G','GGG':'G','GGT':'G'
}

AMINOACIDOS = 'ACDEFGHIKLMNPQRSTVWYX'
CHAR_TO_INT = {char: i for i, char in enumerate(AMINOACIDOS)}
VOCAB_SIZE = len(AMINOACIDOS)
MAX_LEN = 50

# GLC params (example)
GLC_A = 1664525
GLC_C = 1013904223
GLC_M = 2**32
LIMIAR_MUTACAO = 0.02


def gerar_numero_aleatorio(x_n, a=GLC_A, c=GLC_C, m=GLC_M):
    return (a * int(x_n) + c) % m


def escolher_nova_base(original):
    bases = ["A","T","C","G"]
    if original in bases:
        bases.remove(original)
    return random.choice(bases)


def aplicar_mutacoes(sequencia, semente_genetica):
    nova = []
    s = int(semente_genetica) & 0xffffffff
    for base in sequencia:
        s = gerar_numero_aleatorio(s)
        prob = s / GLC_M
        if prob < LIMIAR_MUTACAO:
            nova.append(escolher_nova_base(base))
        else:
            nova.append(base)
    return ''.join(nova)


def traduzir_para_proteina(dna):
    prot = []
    for i in range(0, (len(dna)//3)*3, 3):
        codon = dna[i:i+3]
        aa = CODON_TABLE.get(codon, 'X')
        if aa == '_':
            break
        prot.append(aa)
    return ''.join(prot)


def vetorizar_proteina(proteina_str):
    int_seq = [CHAR_TO_INT.get(ch, VOCAB_SIZE-1) for ch in proteina_str]
    padded = int_seq[:MAX_LEN] + [0]*(MAX_LEN - len(int_seq))
    # one-hot
    arr = np.zeros((1, MAX_LEN, VOCAB_SIZE), dtype=np.float32)
    for i, idx in enumerate(padded):
        if 0 <= idx < VOCAB_SIZE:
            arr[0, i, idx] = 1.0
    return arr


class IAClassi:
    def __init__(self, caminho_modelo=None):
        self.model = None
        if TF_AVAILABLE and caminho_modelo and os.path.exists(caminho_modelo):
            try:
                self.model = keras.models.load_model(caminho_modelo)
            except Exception:
                self.model = None

    def classificar(self, proteina_str):
        if self.model is None:
            # fallback: reproducible pseudo-random score based on sequence
            h = int(hashlib.sha256(proteina_str.encode('utf-8')).hexdigest()[:8], 16)
            return (h % 1000) / 1000.0
        x = vetorizar_proteina(proteina_str)
        pred = self.model.predict(x, verbose=0)
        return float(pred[0][0])

    def classificar_por_contexto(self, proteina_str, cancer_type=""):
        """Classify with optional tumor-type context adjustments.

        For example, for hematological malignancies like 'leucemia' increase
        score for sequences containing specific amino acids (Y/F) that may
        indicate activation sites. This is a simple heuristic overlay on the
        base model score.
        """
        base_score = self.classificar(proteina_str)
        if cancer_type and 'leucemia' in cancer_type.lower():
            # simple heuristic: increase scores when Tyr (Y) or Phe (F) present
            if 'Y' in proteina_str or 'F' in proteina_str:
                base_score = base_score * 1.2
        # clamp to [0,1]
        return min(float(base_score), 1.0)


def criar_e_salvar_modelo_exemplo(caminho_arquivo):
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available in this environment')
    input_shape = (MAX_LEN, VOCAB_SIZE)
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.save(caminho_arquivo)


def _read_first_fasta_sequence(path):
    """Read first sequence (concatenate lines) from a FASTA file."""
    seq = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if seq:
                        break
                    else:
                        continue
                seq.append(line.upper())
        return ''.join(seq)
    except Exception:
        return None


def run_simulation(caminho_modelo_ia, n_mutacoes=100, dna_sequence=None, cancer_type=None):
    ia = IAClassi(caminho_modelo_ia)
    if dna_sequence:
        sequencia_epitopo_dna = dna_sequence
    else:
        sequencia_epitopo_dna = "TGTGCGAGAGATAGCAGCAACTGGTTTGCTTAC"
    semente_paciente = abs(hash(sequencia_epitopo_dna)) % GLC_M
    proteina_original = traduzir_para_proteina(sequencia_epitopo_dna)
    # use context-aware classifier
    if cancer_type:
        impacto_original = ia.classificar_por_contexto(proteina_original, cancer_type=cancer_type)
    else:
        impacto_original = ia.classificar(proteina_original)
    melhor_proteina = proteina_original
    melhor_impacto = impacto_original
    history = []
    for i in range(n_mutacoes):
        dna_mut = aplicar_mutacoes(sequencia_epitopo_dna, semente_paciente + i)
        prot = traduzir_para_proteina(dna_mut)
        if not prot or prot == proteina_original:
            continue
        if cancer_type:
            impacto = ia.classificar_por_contexto(prot, cancer_type=cancer_type)
        else:
            impacto = ia.classificar(prot)
        history.append({'i': i, 'dna': dna_mut, 'prot': prot, 'impacto': impacto})
        if impacto > melhor_impacto:
            melhor_impacto = impacto
            melhor_proteina = prot
    result = {
        'original_protein': proteina_original,
        'original_impact': float(impacto_original),
        'best_protein': melhor_proteina,
        'best_impact': float(melhor_impacto),
        'history': history
    }
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Simular engenharia de anticorpo com IA (exemplo)')
    parser.add_argument('--model', default='modelo_anticorpo_corretivo.keras')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--out-json', type=str, default=None, help='Optional path to write JSON result')
    parser.add_argument('--dna-file', type=str, default=None, help='Path to FASTA file to use as input DNA sequence (first seq)')
    parser.add_argument('--cancer-type', type=str, default=None, help='Optional cancer type context (e.g. leucemia)')
    args = parser.parse_args()
    if args.dna_file:
        dna_seq = _read_first_fasta_sequence(args.dna_file)
        if dna_seq is None:
            print(f'Falha ao ler FASTA: {args.dna_file}')
            dna_seq = None
    else:
        dna_seq = None
    if not os.path.exists(args.model) and TF_AVAILABLE:
        try:
            criar_e_salvar_modelo_exemplo(args.model)
            print('Modelo de exemplo criado:', args.model)
        except Exception as e:
            print('Falha ao criar modelo exemplo:', e)
    res = run_simulation(args.model, n_mutacoes=args.n, dna_sequence=dna_seq, cancer_type=args.cancer_type)
    # write JSON result if requested
    if args.out_json:
        try:
            with open(args.out_json, 'w', encoding='utf-8') as jf:
                json.dump(res, jf, ensure_ascii=False, indent=2)
            print(f'RESULT_JSON: {args.out_json}')
        except Exception as e:
            print(f'Falha ao escrever JSON: {e}')
    # also print a fallback textual representation
    print('Resultado:', res)
