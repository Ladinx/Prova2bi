#Modelos: HistGradientBoosting (HGB) e Random Forest com hiperparametrizacao

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import pandas as pd
import pickle
from pprint import pprint

#   Carregar dados
dados = pd.read_csv("default_of_credit_card_clients.csv", sep=";")
print("Colunas:", list(dados.columns))
print("Shape:", dados.shape)

#   Remover coluna ID (nao relevante para a predicao)
dados = dados.drop(columns=["ID"])

#   Separar atributos e classe
dados_atributos = dados.drop(columns=["default payment next month"])
dados_classes = dados["default payment next month"]

#   Codificar variaveis categoricas (SEX, EDUCATION, MARRIAGE)
#   Usar OrdinalEncoder para manter a ordem ordinal das categorias
cat_cols = dados_atributos.select_dtypes(include=["object", "str"]).columns
print("\nColunas categoricas:", list(cat_cols))
print("Valores unicos por coluna categorica:")
for col in cat_cols:
    print(f"  {col}: {dados_atributos[col].unique()}")

#   Aplicar codificacao ordinal
encoder = OrdinalEncoder(dtype=int)
dados_atributos[cat_cols] = encoder.fit_transform(dados_atributos[cat_cols])

print(f"\nTotal de atributos apos codificacao: {dados_atributos.shape[1]}")
print("Frequencia das classes:\n", dados_classes.value_counts())

#   Balanceamento com SMOTE
print("\n=== Balanceamento com SMOTE ===")
print("Frequencia das classes antes do balanceamento:")
print(dados_classes.value_counts())

balancer = SMOTE(random_state=42)
dados_atributos, dados_classes = balancer.fit_resample(dados_atributos, dados_classes)

print("\nFrequencia das classes apos balanceamento:")
print(dados_classes.value_counts())

#   Separacao treino/teste
atributos_train, atributos_test, classe_train, classes_test = train_test_split(
    dados_atributos, dados_classes, test_size=0.3, random_state=42
)

print(f"\nTamanho treino: {atributos_train.shape[0]}, Tamanho teste: {atributos_test.shape[0]}")


# MODELO 1: HistGradientBoosting (HGB)

print("\n" + "="*60)
print("=== TREINAMENTO: HistGradientBoosting (HGB) ===")
print("="*60)

hgb = HistGradientBoostingClassifier(random_state=42, categorical_features=None)

#Hiperparametrizacao do HGB
print("\n--- Hiperparametrizacao do HGB ---")

#Dominios dos hiperparametros
learning_rate = [float(x) for x in np.linspace(start=0.01, stop=0.3, num=5)]
max_depth = [int(x) for x in np.linspace(start=3, stop=15, num=5)]
max_iter = [int(x) for x in np.linspace(start=50, stop=300, num=6)]
min_samples_leaf = [int(x) for x in np.linspace(start=10, stop=100, num=5)]
l2_regularization = [0.0, 0.1, 0.5, 1.0]

hgb_grid = {
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    'max_iter': max_iter,
    'min_samples_leaf': min_samples_leaf,
    'l2_regularization': l2_regularization
}

hgb_hyperparameters = RandomizedSearchCV(
    estimator=HistGradientBoostingClassifier(random_state=42),
    param_distributions=hgb_grid,
    n_iter=20,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
hgb_hyperparameters.fit(dados_atributos, dados_classes)

pprint("Melhores Hiperparametros para HGB:")
pprint(hgb_hyperparameters.best_params_)

#Modelo HGB otimizado
hgb_melhor = HistGradientBoostingClassifier(**hgb_hyperparameters.best_params_, random_state=42)
hgb_melhor.fit(atributos_train, classe_train)

#   Validacao cruzada do HGB
print("\n=== Validacao Cruzada (HGB Otimizado) ===")
scoring = ["precision_macro", "recall_macro", "f1_macro", "accuracy"]
score_cross_hgb = cross_validate(
    hgb_melhor, dados_atributos, dados_classes, cv=5,
    scoring=scoring, verbose=0, n_jobs=-1
)
print("HGB - Precision:", score_cross_hgb["test_precision_macro"].mean())
print("HGB - Recall:   ", score_cross_hgb["test_recall_macro"].mean())
print("HGB - F1:       ", score_cross_hgb["test_f1_macro"].mean())
print("HGB - Accuracy: ", score_cross_hgb["test_accuracy"].mean())


# MODELO 2: Random Forest (RF) com hiperparametrizacao

print("\n" + "="*60)
print("=== TREINAMENTO: Random Forest (RF) ===")
print("="*60)

#Hiperparametrizacao do Random Forest
print("\n--- Hiperparametrizacao do Random Forest ---")

n_estimators = [int(x) for x in np.linspace(start=50, stop=500, num=10)]
criterion = ["gini", "entropy"]
min_samples_split = [int(x) for x in np.linspace(start=2, stop=20, num=5)]
max_depth = [int(x) for x in np.linspace(start=5, stop=50, num=10)]
max_features = ["sqrt", "log2"]
min_samples_leaf = [int(x) for x in np.linspace(start=1, stop=10, num=4)]

rf_grid = {
    "n_estimators": n_estimators,
    "criterion": criterion,
    "min_samples_split": min_samples_split,
    "max_depth": max_depth,
    "max_features": max_features,
    "min_samples_leaf": min_samples_leaf,
}

rf_hyperparameters = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_grid,
    n_iter=20,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42,
)
rf_hyperparameters.fit(dados_atributos, dados_classes)

pprint("Melhores Hiperparametros para Random Forest:")
pprint(rf_hyperparameters.best_params_)

#Modelo Random Forest otimizado
rf_melhor = RandomForestClassifier(**rf_hyperparameters.best_params_, random_state=42)
rf_melhor.fit(atributos_train, classe_train)

#   Validacao cruzada do Random Forest
print("\n=== Validacao Cruzada (RF Otimizado) ===")
score_cross_rf = cross_validate(
    rf_melhor, dados_atributos, dados_classes, cv=5,
    scoring=scoring, verbose=0, n_jobs=-1
)
print("RF - Precision:", score_cross_rf["test_precision_macro"].mean())
print("RF - Recall:   ", score_cross_rf["test_recall_macro"].mean())
print("RF - F1:       ", score_cross_rf["test_f1_macro"].mean())
print("RF - Accuracy: ", score_cross_rf["test_accuracy"].mean())


# SELECAO DO MELHOR MODELO

print("\n" + "="*60)
print("=== SELECAO DO MELHOR MODELO ===")
print("="*60)

#Comparar F1-score para selecionar o melhor modelo
f1_hgb = score_cross_hgb["test_f1_macro"].mean()
f1_rf = score_cross_rf["test_f1_macro"].mean()

print(f"F1 HGB: {f1_hgb:.4f}")
print(f"F1 RF:  {f1_rf:.4f}")

if f1_hgb >= f1_rf:
    modelo_final = hgb_melhor
    nome_modelo_final = "HistGradientBoosting (HGB)"
    print(f"\nMelhor modelo selecionado: {nome_modelo_final}")
else:
    modelo_final = rf_melhor
    nome_modelo_final = "Random Forest (RF)"
    print(f"\nMelhor modelo selecionado: {nome_modelo_final}")

#   Salvando modelos
print("\n=== Salvando Modelos ===")
pickle.dump(hgb_melhor, open("default_hgb.pkl", "wb"))
pickle.dump(rf_melhor, open("default_rf.pkl", "wb"))
pickle.dump(modelo_final, open("default_melhor_modelo.pkl", "wb"))
pickle.dump(encoder, open("default_encoder.pkl", "wb"))
print("Modelos salvos: default_hgb.pkl, default_rf.pkl, default_melhor_modelo.pkl")
print("Encoder salvo: default_encoder.pkl")


# PREDICOES E METRICAS

print("\n" + "="*60)
print("=== AVALIACAO NO CONJUNTO DE TESTE ===")
print("="*60)

predictions_hgb = hgb_melhor.predict(atributos_test)
predictions_rf = rf_melhor.predict(atributos_test)
predictions_final = modelo_final.predict(atributos_test)

#Probabilidades do modelo final
probabilidades = modelo_final.predict_proba(atributos_test)

print(f"\n--- {nome_modelo_final} (Modelo Final) ---")
print(f"  Acuracia       : {accuracy_score(classes_test, predictions_final):.4f}")
print(f"\nRelatorio de Classificacao:")
print(classification_report(classes_test, predictions_final, target_names=["Nao Default", "Default"]))

#Matriz de confusao - Modelo Final
ConfusionMatrixDisplay.from_estimator(modelo_final, atributos_test, classes_test)
plt.title(f"Matriz de Confusao - {nome_modelo_final}")
plt.savefig("confusionmatrix_default.png")
plt.close()
print("\nMatriz de confusao salva: confusionmatrix_default.png")

#Metricas detalhadas para todos os modelos
for nome, preds in [
    ("HGB", predictions_hgb),
    ("Random Forest", predictions_rf),
    (nome_modelo_final, predictions_final),
]:
    tn, fp, fn, tp = confusion_matrix(classes_test, preds).ravel()
    print(f"\n── {nome} ──")
    print(f"  Acuracia       : {accuracy_score(classes_test, preds):.4f}")
    print(f"  Especificidade : {tn/(tn+fp):.4f}")
    print(f"  Sensibilidade  : {tp/(tp+fn):.4f}")

#   Distribuicao das probabilidades
print("\n" + "="*60)
print("=== DISTRIBUICAO DE PROBABILIDADES ===")
print("="*60)
print(f"\nDistribuicao das probabilidades de default (modelo: {nome_modelo_final}):")
print(f"  Media: {probabilidades[:, 1].mean():.4f}")
print(f"  Mediana: {np.median(probabilidades[:, 1]):.4f}")
print(f"  Desvio Padrao: {probabilidades[:, 1].std():.4f}")
print(f"  Min: {probabilidades[:, 1].min():.4f}")
print(f"  Max: {probabilidades[:, 1].max():.4f}")

#Histograma das probabilidades
plt.figure(figsize=(10, 6))
plt.hist(probabilidades[:, 1], bins=30, alpha=0.7, color="steelblue", edgecolor="black")
plt.axvline(x=0.5, color="red", linestyle="--", label="Limite de decisao (0.5)")
plt.xlabel("Probabilidade de Default")
plt.ylabel("Frequencia")
plt.title(f"Distribuicao de Probabilidades - {nome_modelo_final}")
plt.legend()
plt.savefig("distribuicao_probabilidades.png")
plt.close()
print("Histograma salvo: distribuicao_probabilidades.png")
