# PythonIA

Simulador Volterra-Stieltjes e ferramentas auxiliares para "engenharia" de anticorpos com heurística de IA.

## Foco Técnico

Este projeto, PythonIA, é uma ferramenta de bioinformática computacional que integra simulações de sistemas biológicos complexos com heurísticas de Inteligência Artificial para a otimização e engenharia de anticorpos e proteínas.

O núcleo da simulação é baseado no Integrador Volterra-Stieltjes (V-S), que permite o mapeamento de eventos dinâmicos (como mutações ou interações residuais) ao longo do tempo — essencial para sistemas com memória e dependência de caminho, como dobra e função de proteínas.

- Simulação: `PythonIA/PythonIA.py` implementa o V-S para gerar séries temporais de estados de proteína e um dashboard PNG que resume a dinâmica do sistema.
- Heurística de IA: `PythonIA/simular_anticorpo.py` utiliza classificadores (Keras, se disponível) ou heurísticas determinísticas para pontuar mutações em sequências DNA/proteína e acelerar o ciclo de design.
- Visualização: utilitários como `PythonIA/visualize_p53_r248w.py` mapeiam eventos simulados para resíduos e geram visualizações moleculares (PDB/HTML/3D).

Licença
-------
Este repositório está licenciado sob a MIT License — veja `LICENSE.md`.

Visão geral
-----------
- `PythonIA/PythonIA.py`: simulação Volterra-Stieltjes (geração de dashboard PNG).
- `PythonIA/simular_anticorpo.py`: script de exemplo que aplica mutações a uma sequência DNA e usa um classificador (Keras ou heurística) para pontuar proteínas.
- `PythonIA/run_and_visualize.py`: mapeia eventos da simulação para resíduos e tenta gerar visualização (PDB / HTML).
- `PythonIA/dashboard.py`: aplicativo Streamlit para execução interativa e visualização.
- `PythonIA/visualize_p53_r248w.py`: utilitário para baixar PDB e exportar visualização HTML.
- `PythonIA/utils/`: utilitários para sumarização de FASTA e geração de gráficos.

Requisitos
---------
Recomendado criar ambiente virtual Python3.8+.

Exemplo (PowerShell / Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Se for usar o visualizador3D (py3Dmol) ou TensorFlow, instale-os conforme necessário:

```powershell
python -m pip install py3Dmol
python -m pip install tensorflow # opcional, só se for necessário
```

Como executar
-------------
- Executar simulação principal e gerar dashboard PNG:

```powershell
python PythonIA\PythonIA.py
```

- Rodar simulação de engenharia de anticorpo (usa `simular_anticorpo.py`):

```powershell
python PythonIA\simular_anticorpo.py --model modelo_anticorpo_corretivo.keras --n500 --out-json resultados.json
```

- Executar a interface Streamlit (se quiser):

```powershell
streamlit run PythonIA\dashboard.py
```

## Uso rápido (Windows e Linux/macOS)

A seguir há comandos mínimos para preparar o ambiente, instalar dependências e executar as principais funcionalidades do projeto. Substitua caminhos e nomes de arquivos conforme necessário.

1) Preparar ambiente virtual

- Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

- Linux/macOS (bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Instalar dependências

```bash
pip install -U pip
pip install -r requirements.txt
```

3) Executar a simulação principal (gera PNG de dashboard)

- Windows / Linux:

```bash
python PythonIA/ PythonIA.py
# ou, se preferir caminho Windows:
# python PythonIA\PythonIA.py
```

4) Executar a simulação de engenharia de anticorpo

```bash
python PythonIA/simular_anticorpo.py --model modelo_anticorpo_corretivo.keras --n500 --out-json resultados.json
```

5) Gerar visualização combinada (run_and_visualize)

```bash
python PythonIA/run_and_visualize.py --days365 --threshold0.2
```

6) Interface interativa (Streamlit)

```bash
pip install streamlit
streamlit run PythonIA/dashboard.py
```

7) Notas importantes

- Se usar funcionalidades de ML, instale `tensorflow` apenas se necessário e compatível com seu sistema (CPU/GPU).
- Para visualização3D no navegador sem notebook, a ferramenta `visualize_p53_r248w.py` pode exportar HTML standalone.
- Em Windows, prefira executar os comandos no PowerShell com o ambiente ativado; em Linux/macOS use o bash.
- Se for subir para GitHub, verifique `.gitignore` antes de adicionar arquivos ao repositório.

Esses comandos fornecem um ponto de partida; consulte o README e os comentários nos scripts para parâmetros adicionais e opções avançadas.

Criar repositório Git e subir para GitHub (PowerShell)
----------------------------------------------------
Substitua `SEU_USUARIO` e `SEU_REPO` pelos seus dados. Exemplo usando seu repositório:

```powershell
cd "C:\caminho\para\seu\projeto"
# configurar autor (uma vez)
git config --global user.name "Andre Galberto"
git config --global user.email "seu-email@exemplo.com"

# inicializar e commitar
git init
git add -A
git commit -m "Add project files and MIT license (Andre Galberto,2025)"

git branch -M main
# Usar HTTPS:
git remote add origin https://github.com/angualberto/IAprediction.git
# ou SSH:
# git remote add origin git@github.com:angualberto/IAprediction.git

git push -u origin main
```

Referência / Como citar este trabalho
-----------------------------------
Se este código for usado em pesquisa, cite-o da seguinte forma:

GUALBERTO, André. PythonIA: Simulador Volterra-Stieltjes e ferramentas auxiliares para "engenharia" de anticorpos com heurística de IA. Repositório GitHub. Disponível em: https://github.com/angualberto/IAprediction. Acesso em:29 out.2025.

Observações
-----------
- Substitua `angualberto@gmail.com` pelo seu e-mail real.
- Se usar HTTPS, crie e use um Personal Access Token (PAT) no lugar da senha no primeiro push.

Contato
-------
Andre Galberto — altere `LICENSE.md` para incluir seu e-mail de contato se desejar.
