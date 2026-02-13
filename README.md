# 📈 Dashboard de Momentum - Ações B3

Dashboard interativo para análise de momentum de ações brasileiras, combinando múltiplos indicadores técnicos.

🔗 **[Acesse o Dashboard Online](https://SEU-USUARIO-momentum-dashboard-b3.streamlit.app)** _(atualizar após deploy)_

---

## 🚀 Funcionalidades

### 📊 Indicadores de Momentum
- **Retornos Multi-período**: 1, 3, 6 e 12 meses
- **RSI (Relative Strength Index)**: Identificação de sobrecompra/sobrevenda
- **MACD**: Convergência e divergência de médias móveis
- **Força Relativa vs Ibovespa**: Performance comparada ao índice
- **Volatilidade**: Risco anualizado
- **Score Composto**: Métrica única de 0-100 combinando todos os indicadores

### 📈 Visualizações Interativas
1. **Ranking Completo**: Top 10 e Bottom 10 por momentum
2. **Análise Setorial**: Heatmap e estatísticas por setor
3. **Scatter Risco x Retorno**: Visualização de eficiência
4. **Análise Detalhada**: Drill-down por ação com candlestick e médias móveis

### ⚡ Principais Recursos
- ✅ Análise de ~40 ações mais líquidas da B3
- ✅ Atualização automática dos dados (cache de 2 horas)
- ✅ Filtros por setor e score mínimo
- ✅ Interface responsiva e intuitiva
- ✅ 100% gratuito e open source

---

## 🎯 Como Usar

### Opção 1: Acessar Online (Recomendado)
Acesse diretamente pelo navegador: **[Link do Dashboard](https://SEU-USUARIO-momentum-dashboard-b3.streamlit.app)**

### Opção 2: Rodar Localmente

```bash
# Clone o repositório
git clone https://github.com/arthufvalle/momentum-dashboard-b3.git
cd momentum-dashboard-b3

# Instale as dependências
pip install -r requirements.txt

# Execute o dashboard
streamlit run momentum_dashboard.py
```

O dashboard abrirá em `http://localhost:8501`

---

## 📊 Entendendo o Score de Momentum

O **Score (0-100)** é calculado com os seguintes pesos:

| Componente | Peso | Descrição |
|------------|------|-----------|
| Retorno 6 meses | 25% | Tendência de médio prazo |
| Retorno 3 meses | 20% | Tendência de curto/médio prazo |
| Retorno 1 mês | 15% | Tendência de curto prazo |
| Retorno 12 meses | 10% | Tendência de longo prazo |
| RSI | 10% | Força relativa |
| MACD Histograma | 10% | Convergência/divergência |
| Força Relativa vs Ibovespa | 10% | Performance vs mercado |

### Interpretação dos Scores

- 🟢 **70-100**: Momentum muito forte
- 🟡 **33-70**: Momentum moderado
- 🔴 **0-33**: Momentum fraco

---

## 🛠️ Tecnologias Utilizadas

- **[Streamlit](https://streamlit.io/)**: Framework para interface web
- **[yfinance](https://github.com/ranaroussi/yfinance)**: Dados de mercado
- **[Plotly](https://plotly.com/)**: Gráficos interativos
- **[TA-Lib](https://github.com/bukosabino/ta)**: Indicadores técnicos
- **Pandas & NumPy**: Processamento de dados

---

## 📱 Screenshots

### Ranking de Ações
![Ranking](https://via.placeholder.com/800x400?text=Screenshot+do+Ranking)

### Análise Setorial
![Setorial](https://via.placeholder.com/800x400?text=Screenshot+Análise+Setorial)

### Análise Detalhada
![Detalhada](https://via.placeholder.com/800x400?text=Screenshot+Análise+Detalhada)

---

## 🎓 Casos de Uso

### Para Investidores
- Identificar ações com forte momentum para swing/position trading
- Diversificar por setores com melhor performance
- Monitorar mudanças de tendência

### Para Analistas
- Screening rápido de oportunidades
- Análise setorial comparativa
- Base para análises fundamentalistas complementares

### Para Gestores
- Monitoramento de carteira
- Identificação de rotação setorial
- Gestão de risco baseada em volatilidade

---

## ⚙️ Customização

### Adicionar Mais Ações

Edite a lista `ACOES_B3` no arquivo `momentum_dashboard.py`:

```python
ACOES_B3 = [
    'PETR4.SA', 'VALE3.SA',
    # ... adicione mais tickers aqui
    'NOVO4.SA',  # Seu ticker
]
```

### Ajustar Pesos do Score

Modifique o dicionário `weights` na função `calcular_momentum_score()`:

```python
weights = {
    '21d': 0.15,   # Ajuste conforme sua estratégia
    '63d': 0.20,
    # ...
}
```

---

## ⚠️ Limitações e Disclaimers

- ⚠️ **Este dashboard é apenas para fins educacionais e informativos**
- ⚠️ Não constitui recomendação de investimento
- ⚠️ Dados obtidos via yfinance podem ter delays ou inconsistências
- ⚠️ Performance passada não garante resultados futuros
- ⚠️ Sempre faça sua própria análise antes de investir

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se livre para:

1. Fazer fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abrir um Pull Request

---

## 📝 Roadmap

- [ ] Backtesting automático de estratégias
- [ ] Sistema de alertas (email/Telegram)
- [ ] Exportação para Excel/PDF
- [ ] Integração com dados fundamentalistas
- [ ] Machine Learning para previsão
- [ ] Análise de correlações dinâmicas

---

## 📧 Contato

Criado por **[Seu Nome]** - [@arthufvalle](https://github.com/arthufvalle)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela!**
