import streamlit as st
import pandas as pd
import numpy as np
import requests, json, re
from bs4 import BeautifulSoup
from io import BytesIO
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Análise de Correlação entre Fundos de Investimento")

# Função para buscar dados
def obter_nome_fundo(cnpj):
    cnpj = re.sub(r'\D', '', cnpj)
    url = f"https://www.okanebox.com.br/w/fundo-investimento/{cnpj}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.find('h1').get_text(strip=True)
    else:
        return "Erro ao acessar a página."

def criar_urls(cnpjs):
    fundos = {}
    base_url = "https://www.okanebox.com.br/api/fundoinvestimento/hist/"
    for cnpj in cnpjs:
        cnpj = re.sub(r'\D', '', cnpj)
        nome_fundo = obter_nome_fundo(cnpj)
        url = f"{base_url}{cnpj}/19000101/21000101/"
        fundos[nome_fundo] = url
    return fundos

def coletar_dados_fundos(fundos):
    dados_fundos = {}
    for nome, url in fundos.items():
        response = requests.get(url)
        try:
            dados_historicos = json.loads(response.content.decode("utf-8"))
            df = pd.DataFrame(dados_historicos)
            df['DT_COMPTC'] = pd.to_datetime(df['DT_COMPTC'])
            df.set_index('DT_COMPTC', inplace=True)
            dados_fundos[nome] = df['VL_QUOTA'].pct_change().dropna()
        except:
            continue
    return dados_fundos

def calcular_correlacao(dados_fundos):
    df_completo = pd.concat(dados_fundos, axis=1, join='inner')
    return df_completo.corr()

def plotar_matriz_correlacao(matriz):
    plt.figure(figsize=(10,8))
    sns.heatmap(matriz, annot=True, cmap='Greens', fmt=".2f")
    st.pyplot(plt)

# Entrada de dados no Streamlit
cnpjs_input = st.text_area("Insira até 10 CNPJs, separados por vírgula:")

if st.button("Calcular Correlação"):
    cnpjs = [c.strip() for c in cnpjs_input.split(",") if c.strip()]
    fundos = criar_urls(cnpjs)
    dados = coletar_dados_fundos(fundos)
    matriz = calcular_correlacao(dados)
    st.write("### Matriz de Correlação:")
    plotar_matriz_correlacao(matriz)
