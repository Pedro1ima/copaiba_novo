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

@st.cache_data  # cacheia o resultado para não buscar toda hora
def obter_nome_fundo(cnpj):
    try:
        url = f"https://www.okanebox.com.br/fundo/{cnpj}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        nome = soup.find("h1").get_text(strip=True)
        return nome
    except Exception as e:
        print(f"Erro ao buscar nome do fundo {cnpj}: {e}")
        return f"Fundo {cnpj[-4:]}"


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
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import streamlit as st

    if len(fundos_retornos) < 2:
        st.warning("É necessário pelo menos 2 fundos válidos para calcular correlação.")
        return

    df = pd.concat(fundos_retornos, axis=1, join='inner')
    if df.shape[1] < 2:
        st.warning("Após alinhamento de datas restaram menos de 2 séries válidas para correlacionar.")
        return

    matriz = df.corr()

    # --- Máscara: ocultar somente o triângulo superior (k=1 preserva a diagonal) ---
    mask = np.triu(np.ones_like(matriz, dtype=bool), k=1)

    # --- Criar uma versão anotada onde apenas as células visíveis serão mostradas ---
    anot = matriz.round(2).astype(str)
    anot = anot.where(~mask, "")        # células mascaradas ficam vazias

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(
        matriz,
        mask=mask,
        annot=anot,
        fmt='',
        cmap="Greens",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        annot_kws={"fontsize": 10, "weight": "bold", "color": "white"}
    )

    # Ajustes de ticks para evitar cortar labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # --- Marca d'água (centralizada, sem atrapalhar leitura) ---
    plt.text(
        0.5, 0.5, "Copaíba Invest",
        fontsize=48,
        color="grey",
        alpha=0.18,
        weight="bold",
        ha='center',
        va='center',
        rotation=30,
        transform=ax.transAxes,
        zorder=1  # atrás das células (os patches do heatmap têm zorder maior)
    )

    plt.title("Matriz de Correlação (triângulo inferior com diagonal)")
    plt.tight_layout()
    st.pyplot(plt)


# ---------- UI ----------
st.markdown("Insira até 10 CNPJs, separados por vírgula ou nova linha:")
cnpjs_input = st.text_area("CNPJs", height=120, placeholder="Ex: 13823084000105, 09636393000107, 18860059/0001-15")

   cnpjs = [c.strip() for c in re.split('[,\n;]+', cnpjs_input) if c.strip()]
    if not cnpjs:
        st.error("Por favor, insira pelo menos 1 CNPJ válido.")
    else:
        # --- Mostrar nomes reais dos fundos antes de coletar os dados ---
        st.write("🔍 Buscando nomes dos fundos...")
        nomes_fundos = []
        for c in cnpjs:
            nome = obter_nome_fundo(c)
            nomes_fundos.append(nome)
            st.write(f"- **{nome}**")
        st.divider()

        
        # plotar correlação se possível
        calcular_e_plotar_correlacao(fundos_retornos)

        st.success("Processamento finalizado.")
