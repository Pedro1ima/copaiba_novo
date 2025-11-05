# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import requests, json, re, time
from bs4 import BeautifulSoup
from io import BytesIO
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Correlação Fundos")

st.title("📊 Análise de Correlação entre Fundos de Investimento")

# ---------- Helpers ----------
HEADERS = {
    "Accept-Encoding": "gzip, deflate",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

def clean_cnpj(cnpj_raw):
    if not cnpj_raw:
        return ""
    return re.sub(r"\D", "", cnpj_raw)

def obter_nome_fundo(cnpj):
    """Tenta extrair o nome do fundo da página; retorna None em caso de erro."""
    cnpj = clean_cnpj(cnpj)
    if not cnpj:
        return None
    url = f"https://www.okanebox.com.br/w/fundo-investimento/{cnpj}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.content, "html.parser")
        h1 = soup.find("h1")
        return h1.get_text(strip=True) if h1 else None
    except Exception as e:
        return None

def fetch_json_url(url):
    """Busca a URL e tenta decodificar JSON, suportando gzip."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None, f"Status {r.status_code}"
        # tenta decodificar gzip se houver
        content = r.content
        # algumas respostas podem vir compactadas — requests já lida com isso, mas mantemos tentativa
        try:
            data = json.loads(content.decode("utf-8"))
            return data, None
        except Exception:
            # tentar gunzip
            try:
                buf = BytesIO(content)
                import gzip as _gzip
                with _gzip.GzipFile(fileobj=buf) as f:
                    text = f.read().decode("utf-8")
                    data = json.loads(text)
                    return data, None
            except Exception as e:
                return None, f"JSON decode error: {e}"
    except Exception as e:
        return None, f"Request error: {e}"

def coletar_dados_fundos(cnpjs):
    """
    Retorna dict: {nome_fundo: pd.Series(retornos_diarios)}
    Também retorna lista de erros.
    """
    fundos_retornos = {}
    erros = []
    base_url = "https://www.okanebox.com.br/api/fundoinvestimento/hist/"

    for raw in cnpjs:
        cnpj = clean_cnpj(raw)
        if not cnpj:
            erros.append((raw, "CNPJ inválido após limpeza"))
            continue

        # buscar nome (opcional, pode falhar)
        nome = obter_nome_fundo(cnpj)
        nome_display = nome if nome else f"Fundo_{cnpj[-6:]}"  # fallback

        url = f"{base_url}{cnpj}/19000101/21000101/"
        data_json, err = fetch_json_url(url)
        if err:
            erros.append((cnpj, f"Erro ao buscar dados: {err}"))
            continue
        if not data_json:
            erros.append((cnpj, "Sem dados retornados (lista vazia)"))
            continue

        try:
            df = pd.DataFrame(data_json)
            if 'DT_COMPTC' not in df.columns or 'VL_QUOTA' not in df.columns:
                erros.append((cnpj, "Estrutura JSON inesperada"))
                continue
            df['DT_COMPTC'] = pd.to_datetime(df['DT_COMPTC'])
            df = df.sort_values('DT_COMPTC').set_index('DT_COMPTC')
            # calcular retornos diários
            series_retorno = df['VL_QUOTA'].pct_change().dropna()
            if series_retorno.empty:
                erros.append((cnpj, "Retornos vazios após pct_change"))
                continue
            fundos_retornos[nome_display] = series_retorno
            # evitar rate limit
            time.sleep(0.4)
        except Exception as e:
            erros.append((cnpj, f"Erro ao processar JSON -> DataFrame: {e}"))
            continue

    return fundos_retornos, erros

def calcular_e_plotar_correlacao(fundos_retornos):
    if len(fundos_retornos) < 2:
        st.warning("É necessário pelo menos 2 fundos válidos para calcular correlação.")
        return

    df = pd.concat(fundos_retornos, axis=1, join='inner')
    if df.shape[1] < 2:
        st.warning("Após alinhamento de datas restaram menos de 2 séries válidas para correlacionar.")
        return

    matriz = df.corr()

    # --- Plotar heatmap ---
    plt.figure(figsize=(10,8))
    mask = np.triu(np.ones_like(matriz, dtype=bool), k=1)
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap="Greens", vmin=-1, vmax=1, mask=None)

    # --- Adicionar marca d'água ---
    plt.text(
        0.5, 0.5, "Copaíba Invest",
        fontsize=48, color="white", alpha=0.25, weight="bold",
        ha='center', va='center', rotation=30, transform=plt.gca().transAxes
    )

    plt.tight_layout()
    st.pyplot(plt)

    matriz = df.corr()
    plt.figure(figsize=(10,8))
    mask = np.triu(np.ones_like(matriz, dtype=bool), k=1)
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap="Greens", vmin=-1, vmax=1, mask=None)
    st.pyplot(plt)

# ---------- UI ----------
st.markdown("Insira até 10 CNPJs, separados por vírgula ou nova linha:")
cnpjs_input = st.text_area("CNPJs", height=120, placeholder="Ex: 13823084000105, 09636393000107, 18860059/0001-15")

if st.button("Calcular Correlação"):
    cnpjs = [c.strip() for c in re.split('[,\n;]+', cnpjs_input) if c.strip()]
    if not cnpjs:
        st.error("Por favor, insira pelo menos 1 CNPJ válido.")
    else:
        with st.spinner("Coletando dados..."):
            fundos_retornos, erros = coletar_dados_fundos(cnpjs)

        # mostrar erros (se houver)
        if erros:
            st.error("Alguns CNPJs falharam:")
            for cnpj, msg in erros:
                st.write(f"- {cnpj}: {msg}")

        # mostrar resumo dos fundos capturados
        st.write("### Fundos válidos coletados:")
        for nome, s in fundos_retornos.items():
            st.write(f"- {nome}: {len(s)} observações (de {s.index.min().date()} até {s.index.max().date()})")

        # plotar correlação se possível
        calcular_e_plotar_correlacao(fundos_retornos)

        st.success("Processamento finalizado.")
