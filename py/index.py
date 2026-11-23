import os
import re
from collections import Counter

import requests
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline


PASTA_DADOS = "dados"
PASTA_GRAFICOS = "graficos"

os.makedirs(PASTA_DADOS, exist_ok=True)
os.makedirs(PASTA_GRAFICOS, exist_ok=True)


def coletar_reddit(termo, quantidade=50):

    print(f"\nColetando publicações sobre: {termo}")

    url = "https://www.reddit.com/search.json"

    parametros = {
        "q": termo,
        "limit": min(quantidade, 100),
        "sort": "new",
        "raw_json": 1
    }

    headers = {
        "User-Agent": "ProjetoAcademicoAnaliseSentimentos/1.0"
    }

    resposta = requests.get(
        url,
        params=parametros,
        headers=headers,
        timeout=20
    )

    resposta.raise_for_status()

    dados = resposta.json()

    publicacoes = []

    for item in dados["data"]["children"]:

        postagem = item["data"]

        texto = postagem.get("selftext", "")

        if not texto:
            texto = postagem.get("title", "")

        if texto.strip():

            publicacoes.append({
                "texto": texto,
                "data": pd.to_datetime(
                    postagem.get("created_utc"),
                    unit="s",
                    errors="coerce"
                ),
                "candidato": termo
            })

    return pd.DataFrame(publicacoes)


def limpar_texto(texto):

    texto = str(texto)

    texto = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        texto
    )

    texto = re.sub(
        r"@\w+",
        "",
        texto
    )

    texto = re.sub(
        r"#\w+",
        "",
        texto
    )

    texto = texto.lower()

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


def tratar_dados(df):

    print("\nIniciando limpeza dos dados...")

    df = df.copy()
    
    df = df[
        df["texto"].notna()
    ]

    df = df[
        df["texto"].str.strip() != ""
    ]
    
    df["texto_limpo"] = df["texto"].apply(
        limpar_texto
    )
    
    df = df.drop_duplicates(
        subset=["texto_limpo"]
    )
    
    df = df[
        df["texto_limpo"].str.strip() != ""
    ]

    df = df.reset_index(drop=True)

    print(
        f"Publicações após limpeza: {len(df)}"
    )

    return df



def analisar_sentimentos(df):

    print("\nCarregando modelo de análise de sentimentos...")

    classificador = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )

    sentimentos = []

    for i, texto in enumerate(
        df["texto_limpo"]
    ):

        print(
            f"Analisando publicação {i + 1}/{len(df)}",
            end="\r"
        )

        resultado = classificador(
            texto[:512]
        )[0]

        sentimento = resultado["label"].lower()

        if sentimento == "positive":

            sentimentos.append(
                "Positivo"
            )

        elif sentimento == "negative":

            sentimentos.append(
                "Negativo"
            )

        else:

            sentimentos.append(
                "Neutro"
            )

    df["sentimento"] = sentimentos

    print("\nAnálise concluída!")

    return df



def grafico_sentimentos(df):

    contagem = df[
        "sentimento"
    ].value_counts()

    plt.figure(
        figsize=(8, 5)
    )

    contagem.plot(
        kind="bar"
    )

    plt.title(
        "Distribuição dos Sentimentos"
    )

    plt.xlabel(
        "Sentimento"
    )

    plt.ylabel(
        "Quantidade de Publicações"
    )

    plt.tight_layout()

    plt.savefig(
        f"{PASTA_GRAFICOS}/sentimentos.png",
        dpi=150
    )

    plt.close()



def grafico_candidatos(df):

    tabela = pd.crosstab(
        df["candidato"],
        df["sentimento"]
    )

    tabela.plot(
        kind="bar",
        figsize=(9, 5)
    )

    plt.title(
        "Comparação de Sentimentos por Candidato"
    )

    plt.xlabel(
        "Candidato"
    )

    plt.ylabel(
        "Quantidade de Publicações"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        f"{PASTA_GRAFICOS}/candidatos.png",
        dpi=150
    )

    plt.close()



def grafico_palavras(df):

    palavras = []

    palavras_ignoradas = {
        "que",
        "para",
        "com",
        "uma",
        "não",
        "dos",
        "das",
        "por",
        "como",
        "mais",
        "muito",
        "esse",
        "essa",
        "isso",
        "sobre",
        "seu",
        "sua",
        "eles",
        "elas",
        "tem",
        "foi",
        "ser",
        "está",
        "estão"
    }

    for texto in df["texto_limpo"]:

        palavras_texto = re.findall(
            r"\b[a-záéíóúãõâêôç]{3,}\b",
            texto
        )

        for palavra in palavras_texto:

            if palavra not in palavras_ignoradas:

                palavras.append(
                    palavra
                )

    contagem = Counter(
        palavras
    ).most_common(10)

    if not contagem:
        return

    palavras_df = pd.DataFrame(
        contagem,
        columns=[
            "palavra",
            "frequencia"
        ]
    )

    palavras_df = palavras_df.sort_values(
        "frequencia"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.barh(
        palavras_df["palavra"],
        palavras_df["frequencia"]
    )

    plt.title(
        "Palavras Mais Frequentes"
    )

    plt.xlabel(
        "Frequência"
    )

    plt.tight_layout()

    plt.savefig(
        f"{PASTA_GRAFICOS}/palavras.png",
        dpi=150
    )

    plt.close()



def grafico_evolucao(df):

    dados = df.copy()

    dados["data"] = pd.to_datetime(
        dados["data"],
        errors="coerce"
    )

    dados = dados.dropna(
        subset=["data"]
    )

    if dados.empty:
        print(
            "Não existem datas suficientes para gerar o gráfico."
        )
        return

    quantidade = (
        dados
        .set_index("data")
        .resample("D")
        .size()
    )

    plt.figure(
        figsize=(9, 5)
    )

    quantidade.plot()

    plt.title(
        "Evolução das Manifestações ao Longo do Tempo"
    )

    plt.xlabel(
        "Data"
    )

    plt.ylabel(
        "Quantidade de Publicações"
    )

    plt.tight_layout()

    plt.savefig(
        f"{PASTA_GRAFICOS}/evolucao.png",
        dpi=150
    )

    plt.close()



def gerar_graficos(df):

    print("\nGerando gráficos...")

    grafico_sentimentos(df)

    grafico_candidatos(df)

    grafico_palavras(df)

    grafico_evolucao(df)

    print(
        "Gráficos gerados com sucesso!"
    )



def mostrar_resultados(df):

    total = len(df)

    positivos = (
        df["sentimento"] == "Positivo"
    ).sum()

    negativos = (
        df["sentimento"] == "Negativo"
    ).sum()

    neutros = (
        df["sentimento"] == "Neutro"
    ).sum()

    print("\n")
    print("=" * 50)
    print("RESULTADOS DA ANÁLISE")
    print("=" * 50)

    print(
        f"Total de publicações: {total}"
    )

    print(
        f"Publicações positivas: {positivos}"
    )

    print(
        f"Publicações negativas: {negativos}"
    )

    print(
        f"Publicações neutras: {neutros}"
    )

    if total > 0:

        print()

        print(
            f"Positivas: {(positivos / total) * 100:.2f}%"
        )

        print(
            f"Negativas: {(negativos / total) * 100:.2f}%"
        )

        print(
            f"Neutras: {(neutros / total) * 100:.2f}%"
        )

    print("=" * 50)



def main():

    print("=" * 60)
    print(
        "ANÁLISE DE SENTIMENTOS NAS REDES SOCIAIS"
    )
    print("=" * 60)

    print()

    candidato1 = input(
        "Digite o primeiro candidato: "
    ).strip()

    candidato2 = input(
        "Digite o segundo candidato: "
    ).strip()

    while True:

        try:

            quantidade = int(
                input(
                    "Quantidade de publicações por candidato: "
                )
            )

            if quantidade <= 0:
                print(
                    "Digite uma quantidade maior que zero."
                )
                continue

            break

        except ValueError:

            print(
                "Digite apenas números."
            )

    print()
    print("Iniciando coleta...")

    dados1 = coletar_reddit(
        candidato1,
        quantidade
    )

    dados2 = coletar_reddit(
        candidato2,
        quantidade
    )

    if dados1.empty and dados2.empty:

        print(
            "\nNenhuma publicação foi encontrada."
        )

        return

    dados = pd.concat(
        [dados1, dados2],
        ignore_index=True
    )

    print(
        f"\nTotal coletado: {len(dados)}"
    )
    
    dados.to_csv(
        f"{PASTA_DADOS}/dados_brutos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "Dados brutos salvos em dados/dados_brutos.csv"
    )
    
    dados = tratar_dados(
        dados
    )
    
    dados = analisar_sentimentos(
        dados
    )
    
    dados.to_csv(
        f"{PASTA_DADOS}/dados_analisados.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "Dados analisados salvos em dados/dados_analisados.csv"
    )
    
    gerar_graficos(
        dados
    )
    
    mostrar_resultados(
        dados
    )

    print()
    print(
        "PROJETO FINALIZADO!"
    )



if __name__ == "__main__":
    main()