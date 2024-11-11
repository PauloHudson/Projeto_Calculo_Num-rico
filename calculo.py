import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Carregar dados
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

# Definir pares de colunas para x (variável independente) e y (variável dependente)
pares_de_colunas = [
    ('Área (m²)', 'Valor do Aluguel (R$)'),
    ('Quartos', 'Valor Total (R$)'),
    ('Seguro Incêndio (R$)', 'Valor do Aluguel (R$)'),
    ('Condomínio (R$)', 'Valor do Aluguel (R$)'),
    ('Área (m²)', 'IPTU (R$)')
]

# Executar regressões lineares e exibir resultados
for i, (coluna_x, coluna_y) in enumerate(pares_de_colunas, start=1):
    # Filtrar dados para remover outliers
    dados_filtrados = dados[(dados[coluna_x] < dados[coluna_x].quantile(0.99)) & 
                            (dados[coluna_y] < dados[coluna_y].quantile(0.99))]
    
    x = dados_filtrados[coluna_x]
    y = dados_filtrados[coluna_y]
    
    # Ajustar dados para o formato necessário
    X = np.array(x).reshape(-1, 1)
    Y = y.to_numpy()
    
    # Criar e ajustar o modelo de regressão linear
    modelo = LinearRegression().fit(X, Y)
    y_pred = modelo.predict(X)
    
    # Calcular coeficientes e métricas
    intercepto = modelo.intercept_
    coeficiente = modelo.coef_[0]
    R2 = r2_score(Y, y_pred)
    correlacao_pearson, _ = pearsonr(X.flatten(), Y)
    
    # Exibir resultados
    print(f"\nRegressão {i}: {coluna_x} vs {coluna_y}")
    print(f"Intercepto: {intercepto:.2f}")
    print(f"Coeficiente: {coeficiente:.2f}")
    print(f"Coef. de determinação (R²): {R2:.4f}")
    print(f"Coef. de Pearson: {correlacao_pearson:.4f}")
    
    # Plotar gráfico de dispersão com linha de regressão
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color="blue", s=10, alpha=0.5, label="Dados reais")
    plt.plot(X, y_pred, color="red", linewidth=2, label="Linha de regressão")
    
    # Ajustar escala logarítmica, se necessário
    if X.min() > 0 and Y.min() > 0:  # Verifica se os valores são maiores que zero
        if (X.max() / X.min() > 100) or (Y.max() / Y.min() > 100):  # Condição para diferença significativa de escala
            plt.xscale('log')
            plt.yscale('log')
    
    plt.xlabel(coluna_x)
    plt.ylabel(coluna_y)
    plt.title(f"Regressão {i}: {coluna_x} vs {coluna_y}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
