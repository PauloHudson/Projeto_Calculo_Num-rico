import pandas as pd
import statsmodels.api as sm

# Carregar os dados
dados = pd.read_csv('./houses_to_rent_v2.csv', header=0)

# Renomear as colunas para português
dados.rename(columns={
    'area': 'Área (m²)',
    'rent amount (R$)': 'Valor do Aluguel (R$)',
    'rooms': 'Quartos',
    'total (R$)': 'Valor Total (R$)',
    'bathroom': 'Banheiros',
    'hoa (R$)': 'Condomínio (R$)',
    'parking spaces': 'Vagas de Garagem',
    'property tax (R$)': 'IPTU (R$)',
    'fire insurance (R$)': 'Seguro Incêndio (R$)'
}, inplace=True)

# Filtragem de outliers (Percentil 99)
# Aplicar filtragem em cada variável independente
for coluna in ['Área (m²)', 'Quartos', 'Banheiros', 'Condomínio (R$)']:
    limiar = dados[coluna].quantile(0.99)
    dados = dados[dados[coluna] <= limiar]

# Definir variáveis independentes (X) e dependente (y)
X = dados[['Área (m²)', 'Quartos', 'Banheiros', 'Condomínio (R$)']]
y = dados['Valor Total (R$)']

# Adicionar uma constante para o modelo (intercepto)
X = sm.add_constant(X)

# Criar e ajustar o modelo de regressão linear múltipla (Modelo Completo)
modelo_completo = sm.OLS(y, X).fit()

# Exibir resultados do modelo completo
print("Resultados da Regressão Linear Múltipla (Modelo Completo):")
print(modelo_completo.summary())

# Selecionar apenas variáveis significativas (p-valor < 0.05)
variaveis_significativas = [col for col, p_val in modelo_completo.pvalues.items() if p_val < 0.05 and col != 'const']

# Criar o modelo simplificado usando apenas variáveis significativas
X_significativas = dados[variaveis_significativas]
X_significativas = sm.add_constant(X_significativas)  # Adicionar a constante novamente
modelo_simplificado = sm.OLS(y, X_significativas).fit()

# Exibir resultados do modelo simplificado
print("\nResultados da Regressão Linear Múltipla (Apenas Variáveis Significativas):")
print(modelo_simplificado.summary())
