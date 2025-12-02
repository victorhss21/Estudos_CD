# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# REGRESSÃO LOGÍSTICA
# ==============================================================================
# ==============================================================================
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ==============================================================================
# 1. GERAÇÃO DE DADOS SINTÉTICOS (CENÁRIO REALISTA) - MANTIDO
# ==============================================================================
np.random.seed(42)

def gerar_dados(n_novo=500, n_benchmark=2000):
    df_novo = pd.DataFrame({
        'id_cliente': [f'N_{i}' for i in range(n_novo)],
        'segmento_novo': 1,
        'ramo': np.random.choice(['Varejo', 'Servicos', 'Tecnologia'], n_novo, p=[0.3, 0.2, 0.5]),
        'modelo_op': np.random.choice(['B2B', 'B2C', 'B2B2C'], n_novo, p=[0.4, 0.5, 0.1]),
        'digitalizacao': np.random.normal(85, 10, n_novo).clip(0, 100),
        'tempo_vida_anos': np.random.uniform(0.5, 3, n_novo),
        'faturamento_mensal': np.random.lognormal(10, 0.5, n_novo)
    })
    df_bench = pd.DataFrame({
        'id_cliente': [f'B_{i}' for i in range(n_benchmark)],
        'segmento_novo': 0,
        'ramo': np.random.choice(['Varejo', 'Servicos', 'Tecnologia', 'Industria'], n_benchmark, p=[0.3, 0.3, 0.2, 0.2]),
        'modelo_op': np.random.choice(['B2B', 'B2C', 'B2B2C'], n_benchmark, p=[0.5, 0.4, 0.1]),
        'digitalizacao': np.random.normal(60, 20, n_benchmark).clip(0, 100),
        'tempo_vida_anos': np.random.uniform(5, 20, n_benchmark),
        'faturamento_mensal': np.random.lognormal(12, 1.0, n_benchmark),
        'crescimento_anual_real': np.random.normal(0.15, 0.05, n_benchmark)
    })
    return pd.concat([df_novo, df_bench], ignore_index=True)

df = gerar_dados()

# ==============================================================================
# 2. PREPARAÇÃO E GERAÇÃO DE SCORES COM CROSS-VALIDATION
# ==============================================================================

cat_features = ['ramo', 'modelo_op']
num_features = ['faturamento_mensal', 'digitalizacao']

X = df[cat_features + num_features]
y = df['segmento_novo']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

model_base = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
])

# Seleção Automática do Método de Calibração
calibration_method = 'isotonic' if len(X) > 1000 else 'sigmoid'
print(f"Método de calibração escolhido: {calibration_method.upper()}")

# CONFIGURAÇÃO DO CROSS-VALIDATION
# Usamos CV=5 para gerar scores "out-of-fold" para todo o dataset.
# Isso substitui a divisão fixa Train/Val, aproveitando 100% dos dados para treino e previsão de forma rotativa.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definimos o CalibratedClassifierCV com cv interno para ajustar a calibração dentro de cada fold
calibrated_model_cv = CalibratedClassifierCV(estimator=model_base, method=calibration_method, cv=3) 

# GERAÇÃO DOS SCORES (OOF - Out of Fold Predictions)
# cross_val_predict treina em 4 folds e prevê no 5º, repetindo até cobrir todos.
# Isso garante que o score de cada cliente foi gerado por um modelo que NÃO viu aquele cliente no treino.
y_proba_cv = cross_val_predict(calibrated_model_cv, X, y, cv=cv, method='predict_proba')

# Atribuímos a probabilidade da classe 1 (Novo Segmento)
df['propensity_score'] = y_proba_cv[:, 1]

# --- Verificação Visual: Curva de Calibração (Reliability Diagram) ---
# Como os scores foram gerados via CV, podemos plotar a curva usando todo o dataset (y) sem risco de overfitting
prob_true, prob_pred = calibration_curve(y, df['propensity_score'], n_bins=10)

plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], "k:", label="Perfeitamente Calibrado")
plt.plot(prob_pred, prob_true, "s-", label=f"Modelo Calibrado via CV ({calibration_method})")
plt.ylabel("Fração de Positivos (Real)")
plt.xlabel("Probabilidade Média Predita")
plt.title("Curva de Calibração (Reliability Diagram) - Cross Validation")
plt.legend()
plt.show()

# ==============================================================================
# 3. MATCHING (MANTIDO IGUAL)
# ==============================================================================

epsilon = 1e-6
df['propensity_score_clipped'] = df['propensity_score'].clip(epsilon, 1-epsilon)
df['propensity_logit'] = np.log(df['propensity_score_clipped'] / (1 - df['propensity_score_clipped']))

df_treatment = df[df['segmento_novo'] == 1].copy() 
df_control = df[df['segmento_novo'] == 0].copy()   

knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
knn.fit(df_control[['propensity_logit']])

distances, indices = knn.kneighbors(df_treatment[['propensity_logit']])

matched_indices = df_control.iloc[indices.flatten()].index
df_matched_control = df.loc[matched_indices].reset_index(drop=True)

resultado_matching = pd.DataFrame({
    'ID_Novo': df_treatment['id_cliente'].values,
    'Ramo_Novo': df_treatment['ramo'].values,
    'Score_Calibrado_Novo': df_treatment['propensity_score'].values,
    'ID_Benchmark': df_matched_control['id_cliente'].values,
    'Ramo_Benchmark': df_matched_control['ramo'].values,
    'Score_Calibrado_Benchmark': df_matched_control['propensity_score'].values,
    'Tempo_Vida_Benchmark': df_matched_control['tempo_vida_anos'].values,
    'Meta_Crescimento_Sugerida': df_matched_control['crescimento_anual_real'].values
})

# ==============================================================================
# 4. ANÁLISE E RECOMENDAÇÃO FINAL (MANTIDO)
# ==============================================================================

print("
=== Exemplo de Pares Encontrados (Com Calibração via CV) ===")
print(resultado_matching[['ID_Novo', 'Score_Calibrado_Novo', 'ID_Benchmark', 'Score_Calibrado_Benchmark', 'Meta_Crescimento_Sugerida']].head())

media_crescimento_projetada = resultado_matching['Meta_Crescimento_Sugerida'].mean()
print(f"
=== Resultado da Projeção ===")
print(f"Média de Crescimento Histórico do Grupo Benchmark (Geral): {df_control['crescimento_anual_real'].mean():.2%}")
print(f"Meta de Crescimento Ajustada (Pós-Matching Calibrado): {media_crescimento_projetada:.2%}")

plt.figure(figsize=(10, 6))
sns.kdeplot(df_treatment['propensity_score'], label='Novo Segmento (Alvo)', fill=True, alpha=0.3)
sns.kdeplot(df_control['propensity_score'], label='Outros Segmentos (Fonte)', fill=True, alpha=0.3)
plt.title('Sobreposição dos Scores de Propensão Calibrados via CV (Overlap)')
plt.xlabel('Probabilidade Calibrada de ser do Novo Segmento')
plt.legend()
plt.show()

# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# KERNEL MATCHING
# ==============================================================================
# ==============================================================================
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# ==============================================================================
# 1. GERAÇÃO DE DADOS SINTÉTICOS (MANTIDO IGUAL)
# ==============================================================================
np.random.seed(42)

def gerar_dados(n_novo=500, n_benchmark=2000):
    # --- Grupo Novo Segmento (Alvo) ---
    df_novo = pd.DataFrame({
        'id_cliente': [f'N_{i}' for i in range(n_novo)],
        'segmento_novo': 1,
        'ramo': np.random.choice(['Varejo', 'Servicos', 'Tecnologia'], n_novo, p=[0.3, 0.2, 0.5]),
        'modelo_op': np.random.choice(['B2B', 'B2C', 'B2B2C'], n_novo, p=[0.4, 0.5, 0.1]),
        'digitalizacao': np.random.normal(85, 10, n_novo).clip(0, 100),
        'tempo_vida_anos': np.random.uniform(0.5, 3, n_novo),
        'faturamento_mensal': np.random.lognormal(10, 0.5, n_novo)
    })

    # --- Grupo Benchmark (Outros Segmentos) ---
    df_bench = pd.DataFrame({
        'id_cliente': [f'B_{i}' for i in range(n_benchmark)],
        'segmento_novo': 0,
        'ramo': np.random.choice(['Varejo', 'Servicos', 'Tecnologia', 'Industria'], n_benchmark, p=[0.3, 0.3, 0.2, 0.2]),
        'modelo_op': np.random.choice(['B2B', 'B2C', 'B2B2C'], n_benchmark, p=[0.5, 0.4, 0.1]),
        'digitalizacao': np.random.normal(60, 20, n_benchmark).clip(0, 100),
        'tempo_vida_anos': np.random.uniform(5, 20, n_benchmark),
        'faturamento_mensal': np.random.lognormal(12, 1.0, n_benchmark),
        'crescimento_anual_real': np.random.normal(0.15, 0.05, n_benchmark)
    })

    return pd.concat([df_novo, df_bench], ignore_index=True)

df = gerar_dados()

# ==============================================================================
# 2. PREPARAÇÃO E SCORE (COM CROSS-VALIDATION)
# ==============================================================================
cat_features = ['ramo', 'modelo_op']
num_features = ['faturamento_mensal', 'digitalizacao']

X = df[cat_features + num_features]
y = df['segmento_novo']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

base_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
])

# 2.1 Configuração da Calibração e Validação Cruzada
metodo_calibracao = 'isotonic' if len(df) > 1000 else 'sigmoid'
print(f"Aplicando calibração via método: {metodo_calibracao.upper()}")

# MODELO INTERNO (CV=3):
# O CalibratedClassifierCV já faz uma validação interna para ajustar a curva de calibração.
# Ao definir cv=3 aqui, ele divide o treino em 3 folds para calibrar.
calibrated_clf = CalibratedClassifierCV(base_clf, method=metodo_calibracao, cv=3)

# GERAÇÃO DE SCORES VIA CROSS-VALIDATION EXTERNO (CV=5):
# Aqui está a melhoria. Em vez de fit(X,y) -> predict(X), usamos cross_val_predict.
# Isso garante que o score de cada cliente seja gerado por um modelo que não viu aquele cliente.
# cv=5 é um padrão robusto. method='predict_proba' retorna as probabilidades.
cv_scores = cross_val_predict(calibrated_clf, X, y, cv=5, method='predict_proba')

# Pegamos a probabilidade da classe positiva (1)
df['propensity_score'] = cv_scores[:, 1]

# Treinamos o modelo final em TODOS os dados apenas para análise futura de features se necessário,
# mas os scores usados para matching são os validados (cv_scores).
calibrated_clf.fit(X, y) 

# ==============================================================================
# 2.2 VISUALIZAÇÃO DA CURVA DE CALIBRAÇÃO (MANTIDO)
# ==============================================================================
prob_true, prob_pred = calibration_curve(y, df['propensity_score'], n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfeitamente Calibrado')
plt.plot(prob_pred, prob_true, marker='.', label=f'Modelo Calibrado ({metodo_calibracao}) - Via CV')
plt.xlabel('Probabilidade Predita Média')
plt.ylabel('Fração de Positivos (Real)')
plt.title('Curva de Calibração (Reliability Diagram)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Clip e Logit (MANTIDO)
eps = 1e-6
df['propensity_score_clipped'] = df['propensity_score'].clip(eps, 1 - eps)
df['propensity_logit'] = np.log(df['propensity_score_clipped'] / (1 - df['propensity_score_clipped']))

# ==============================================================================
# 3. REALIZANDO O KERNEL MATCHING (MANTIDO)
# ==============================================================================

# Separar os grupos (importante resetar index para alinhar com arrays numpy)
df_treatment = df[df['segmento_novo'] == 1].copy().reset_index(drop=True)
df_control = df[df['segmento_novo'] == 0].copy().reset_index(drop=True)

T_logits = df_treatment['propensity_logit'].values
C_logits = df_control['propensity_logit'].values
C_outcomes = df_control['crescimento_anual_real'].values 

bandwidth = 0.25 

def gaussian_kernel(distance, h):
    return (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-0.5 * (distance / h) ** 2)

projections = []
weights_sum_list = []

print("Iniciando Kernel Matching...")

for t_logit in T_logits:
    diffs = t_logit - C_logits
    weights = gaussian_kernel(diffs, bandwidth)
    sum_weights = np.sum(weights)
    weights_sum_list.append(sum_weights)
    
    if sum_weights < 1e-4: 
        projections.append(np.nan)
    else:
        weighted_outcome = np.sum(weights * C_outcomes) / sum_weights
        projections.append(weighted_outcome)

df_treatment['Meta_Crescimento_Sugerida_Kernel'] = projections
df_treatment['Peso_Total_Encontrado'] = weights_sum_list 

n_sem_match = df_treatment['Meta_Crescimento_Sugerida_Kernel'].isna().sum()
print(f"Clientes sem match suficiente (Discarded): {n_sem_match} de {len(df_treatment)}")

resultado_matching = df_treatment.dropna(subset=['Meta_Crescimento_Sugerida_Kernel'])

# ==============================================================================
# 4. ANÁLISE E RECOMENDAÇÃO FINAL
# ==============================================================================

print("=== Exemplo de Resultados (Kernel Matching) ===")
print(resultado_matching[['id_cliente', 'ramo', 'propensity_score', 'Meta_Crescimento_Sugerida_Kernel']].head())

media_crescimento_projetada = resultado_matching['Meta_Crescimento_Sugerida_Kernel'].mean()
print(f"
=== Resultado da Projeção ===")
print(f"Média de Crescimento Histórico do Grupo Benchmark (Geral): {df_control['crescimento_anual_real'].mean():.2%}")
print(f"Meta de Crescimento Ajustada (Kernel Matching): {media_crescimento_projetada:.2%}")

plt.figure(figsize=(10, 6))
sns.kdeplot(df_treatment['propensity_score'], label='Novo Segmento (Alvo)', fill=True, alpha=0.3)
sns.kdeplot(df_control['propensity_score'], label='Outros Segmentos (Fonte)', fill=True, alpha=0.3)
plt.title('Sobreposição dos Scores de Propensão (Overlap) - Gerado via CV')
plt.xlabel('Probabilidade de ser do Novo Segmento')
plt.legend()
plt.show()
