#Modulo de Inferencia - Previsao de Inadimplencia (Default)
#Carrega o melhor modelo e faz predicoes para novas instancias
#Retorna: classe predita + distribuicao de probabilidades

from pickle import load
import numpy as np
import pandas as pd
from pprint import pprint

#   Carregar o melhor modelo e o encoder
modelo = load(open('default_melhor_modelo.pkl', 'rb'))
encoder = load(open('default_encoder.pkl', 'rb'))

print("="*60)
print("MODULO DE INFERENCIA - Previsao de Inadimplencia")
print("="*60)

#   Nova instancia para predicao
#   Formato: [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
#             PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
#             BILL_AMT1..6, PAY_AMT1..6]
#
#   Exemplo: cliente com limite 50000, masculino, ensino medio, casado, 35 anos,
#            pagamentos em dia (PAY_0=0), faturas e pagamentos diversos

novas_instancias = [
    #Cliente 1: Baixo risco - pagamentos em dia, limite moderado
    [50000, "M", "Middle School ", "Married", 35,
     0, 0, 0, 0, 0, 0,
     30000, 28000, 25000, 22000, 20000, 18000,
     5000, 5000, 5000, 5000, 5000, 5000],

    #Cliente 2: Alto risco - atrasos frequentes, limite baixo
    [20000, "F", "High School ", "Divorced", 28,
     2, 3, 2, 1, 2, 2,
     15000, 18000, 16000, 14000, 12000, 10000,
     500, 0, 300, 0, 200, 0],

    #Cliente 3: Risco moderado - alguns atrasos
    [80000, "M", "Post-Secondary Non-Tertiary Education", "Married", 42,
     -1, 0, -1, 0, 0, 0,
     60000, 55000, 50000, 48000, 45000, 42000,
     10000, 8000, 9000, 7000, 8000, 7500],
]

#   Converter para DataFrame e aplicar o mesmo encoder do treinamento
colunas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]

for i, instancia in enumerate(novas_instancias):
    print(f"\n{'='*60}")
    print(f"CLIENTE {i+1}")
    print(f"{'='*60}")

    #Criar DataFrame com a mesma estrutura usada no treinamento
    colunas = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
               "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
               "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4",
               "BILL_AMT5", "BILL_AMT6",
               "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4",
               "PAY_AMT5", "PAY_AMT6"]

    df_instancia = pd.DataFrame([instancia], columns=colunas)

    #Aplicar o encoder nas colunas categoricas
    df_instancia[colunas_categoricas] = encoder.transform(df_instancia[colunas_categoricas])

    #   Predicao da classe (0 = Nao Default, 1 = Default)
    classe_predita = modelo.predict(df_instancia)[0]
    resultado = "INADIMPLENTE (DEFAULT)" if classe_predita == 1 else "NAO INADIMPLENTE"

    #   Distribuicao de probabilidades [P(nao_default), P(default)]
    probabilidades = modelo.predict_proba(df_instancia)[0]
    prob_default = probabilidades[1] * 100
    prob_nao_default = probabilidades[0] * 100

    print(f"  Classe Predita        : {classe_predita} -> {resultado}")
    print(f"\n  Distribuicao de Probabilidades:")
    print(f"    Probabilidade de Default     : {prob_default:.2f}%")
    print(f"    Probabilidade de Nao Default : {prob_nao_default:.2f}%")

    #Score de risco (0-100)
    score_risco = prob_default
    if score_risco >= 70:
        nivel_risco = "ALTO"
    elif score_risco >= 30:
        nivel_risco = "MODERADO"
    else:
        nivel_risco = "BAIXO"

    print(f"\n  Score de Risco        : {score_risco:.2f} / 100")
    print(f"  Nivel de Risco        : {nivel_risco}")
    print(f"  Decisao               : {'NEGAR CREDITO' if classe_predita == 1 else 'APROVAR CREDITO'}")

    print(f"\n  Dados do Cliente:")
    print(f"    Limite:         R$ {instancia[0]:,.2f}")
    print(f"    Sexo:           {instancia[1]}")
    print(f"    Educacao:       {instancia[2]}")
    print(f"    Estado Civil:   {instancia[3]}")
    print(f"    Idade:          {instancia[4]}")

print(f"\n{'='*60}")
print("INFERENCIA CONCLUIDA")
print(f"{'='*60}")
