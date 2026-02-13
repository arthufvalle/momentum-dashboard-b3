"""
Dashboard de Índice de Momentum - Ações Brasileiras (B3)
Análise multifatorial de momentum com visualizações interativas
Versão 2.0 - Atualizada com setores corretos e aba de reversão
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from ta.momentum import RSIIndicator
from ta.trend import MACD
import time

# Configuração da página
st.set_page_config(
    page_title="Momentum Dashboard - B3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# Carregar ações e setores do CSV (inline para não depender de arquivo externo)
ACOES_SETORES = {
    'ABEV3.SA': 'Alimentos e Bebidas',
    'ALOS3.SA': 'Shoppings',
    'ALPA4.SA': 'Comércio Varejista',
    'ALUP11.SA': 'Energia Elétrica',
    'AMBP3.SA': 'Energia Elétrica',
    'AMER3.SA': 'E-Commerce',
    'ANIM3.SA': 'Educação',
    'AZZA3.SA': 'Comércio Varejista',
    'ASAI3.SA': 'Supermercados',
    'AURE3.SA': 'Energia Elétrica',
    'B3SA3.SA': 'Financeiros',
    'BBAS3.SA': 'Financeiros',
    'BBDC3.SA': 'Financeiros',
    'BBDC4.SA': 'Financeiros',
    'BBSE3.SA': 'Financeiros',
    'BEEF3.SA': 'Alimentos e Bebidas',
    'BHIA3.SA': 'E-Commerce',
    'BIDI4.SA': 'Financeiros',
    'BPAC11.SA': 'Financeiros',
    'BPAN4.SA': 'Financeiros',
    'BRAP4.SA': 'Mineração e Siderurgia',
    'BRAV3.SA': 'Petróleo e Gás',
    'VBBR3.SA': 'Distribuição de Combustíveis',
    'BRFS3.SA': 'Alimentos e Bebidas',
    'BRKM5.SA': 'Mineração e Siderurgia',
    'CAML3.SA': 'Alimentos e Bebidas',
    'CASH3.SA': 'TMT',
    'CBAV3.SA': 'Mineração e Siderurgia',
    'CCRO3.SA': 'Infraestrutura',
    'CEAB3.SA': 'Comércio Varejista',
    'CMIG4.SA': 'Energia Elétrica',
    'CMIN3.SA': 'Mineração e Siderurgia',
    'COGN3.SA': 'Educação',
    'CPFE3.SA': 'Energia Elétrica',
    'CPLE3.SA': 'Energia Elétrica',
    'CSAN3.SA': 'Distribuição de Combustíveis',
    'CSMG3.SA': 'Água e Saneamento',
    'CSNA3.SA': 'Mineração e Siderurgia',
    'CURY3.SA': 'Construção Civil',
    'CVCB3.SA': 'Comércio Varejista',
    'CXSE3.SA': 'Financeiros',
    'CYRE3.SA': 'Construção Civil',
    'DIRR3.SA': 'Energia Elétrica',
    'DXCO3.SA': 'Madeira e Papel',
    'ECOR3.SA': 'Infraestrutura',
    'EGIE3.SA': 'Energia Elétrica',
    'ELET3.SA': 'Energia Elétrica',
    'EMBR3.SA': 'Bens Industriais',
    'ENBR3.SA': 'Energia Elétrica',
    'ENEV3.SA': 'Energia Elétrica',
    'ENGI4.SA': 'Energia Elétrica',
    'EQTL3.SA': 'Energia Elétrica',
    'EVEN3.SA': 'Construção Civil',
    'EZTC3.SA': 'Construção Civil',
    'FLRY3.SA': 'Saúde',
    'GGBR4.SA': 'Mineração e Siderurgia',
    'GMAT3.SA': 'Supermercados',
    'GOAU4.SA': 'Mineração e Siderurgia',
    'GOLL4.SA': 'Linhas Aéreas',
    'HAPV3.SA': 'Saúde',
    'HYPE3.SA': 'Saúde',
    'IGTI11.SA': 'Shoppings',
    'IRBR3.SA': 'Financeiros',
    'ITSA4.SA': 'Financeiros',
    'ITUB4.SA': 'Financeiros',
    'JBSS3.SA': 'Alimentos e Bebidas',
    'JHSF3.SA': 'Construção Civil',
    'KLBN11.SA': 'Madeira e Papel',
    'LREN3.SA': 'Comércio Varejista',
    'MGLU3.SA': 'E-Commerce',
    'MOVI3.SA': 'Aluguel de Carros',
    'MRVE3.SA': 'Construção Civil',
    'MULT3.SA': 'Shoppings',
    'PCAR3.SA': 'Supermercados',
    'PETR3.SA': 'Petróleo e Gás',
    'PETR4.SA': 'Petróleo e Gás',
    'PRIO3.SA': 'Petróleo e Gás',
    'RADL3.SA': 'Comércio Varejista',
    'RAIL3.SA': 'Infraestrutura',
    'RAIZ4.SA': 'Distribuição de Combustíveis',
    'RDOR3.SA': 'Saúde',
    'RECV3.SA': 'Petróleo e Gás',
    'RENT3.SA': 'Aluguel de Carros',
    'SANB11.SA': 'Financeiros',
    'SAPR11.SA': 'Água e Saneamento',
    'SBSP3.SA': 'Água e Saneamento',
    'SUZB3.SA': 'Madeira e Papel',
    'TAEE11.SA': 'Energia Elétrica',
    'TIMS3.SA': 'TMT',
    'TOTS3.SA': 'TMT',
    'TUPY3.SA': 'Mineração e Siderurgia',
    'UGPA3.SA': 'Distribuição de Combustíveis',
    'USIM5.SA': 'Mineração e Siderurgia',
    'VALE3.SA': 'Mineração e Siderurgia',
    'VAMO3.SA': 'Aluguel de Carros',
    'VIVT3.SA': 'TMT',
    'WEGE3.SA': 'Bens Industriais',
    'YDUQ3.SA': 'Educação',
}

ACOES_B3 = list(ACOES_SETORES.keys())


@st.cache_data(ttl=7200)  # Cache por 2 horas
def download_data(tickers, period='1y'):
    """Download dados históricos com cache e retry"""
    data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                status_text.text(f"Baixando {ticker}... ({idx+1}/{len(tickers)})")
                df = yf.download(ticker, period=period, progress=False, timeout=10)
                
                # Normalizar colunas se vier com multi-index
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if len(df) > 20:  # Mínimo de dados
                    data[ticker] = df
                    break
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(1)
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    pass  # Silenciar warnings para não poluir
                else:
                    time.sleep(1)
        
        progress_bar.progress((idx + 1) / len(tickers))
    
    progress_bar.empty()
    status_text.empty()
    
    # Download do índice de referência
    try:
        ibov = yf.download('^BVSP', period=period, progress=False, timeout=10)
        if isinstance(ibov.columns, pd.MultiIndex):
            ibov.columns = ibov.columns.get_level_values(0)
    except:
        ibov = pd.DataFrame()
    
    return data, ibov


def calcular_retornos(df, periodos=[21, 63, 126, 252]):
    """Calcula retornos em múltiplos períodos (dias úteis)"""
    retornos = {}
    
    for periodo in periodos:
        if len(df) >= periodo:
            ret = (df['Close'].iloc[-1] / df['Close'].iloc[-periodo] - 1) * 100
            retornos[f'{periodo}d'] = ret
        else:
            retornos[f'{periodo}d'] = np.nan
    
    return retornos


def calcular_rsi(df, period=14):
    """Calcula RSI"""
    if len(df) < period:
        return np.nan
    
    try:
        rsi = RSIIndicator(close=df['Close'], window=period)
        return rsi.rsi().iloc[-1]
    except:
        return np.nan


def calcular_macd(df):
    """Calcula MACD e retorna o histograma"""
    if len(df) < 26:
        return np.nan, np.nan
    
    try:
        macd = MACD(close=df['Close'])
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        return macd_line, macd_line - signal_line
    except:
        return np.nan, np.nan


def calcular_forca_relativa(df_ativo, df_indice, periodo=126):
    """Calcula força relativa vs índice"""
    if len(df_ativo) < periodo or len(df_indice) < periodo:
        return np.nan
    
    try:
        ret_ativo = (df_ativo['Close'].iloc[-1] / df_ativo['Close'].iloc[-periodo] - 1)
        ret_indice = (df_indice['Close'].iloc[-1] / df_indice['Close'].iloc[-periodo] - 1)
        return (ret_ativo - ret_indice) * 100
    except:
        return np.nan


def calcular_volatilidade(df, periodo=21):
    """Calcula volatilidade anualizada"""
    if len(df) < periodo:
        return np.nan
    
    try:
        returns = df['Close'].pct_change().dropna()
        vol = returns.tail(periodo).std() * np.sqrt(252) * 100
        return vol
    except:
        return np.nan


def calcular_momentum_score(metrics):
    """Calcula score composto de momentum (0-100)"""
    score = 0
    weights = {
        '21d': 0.15,
        '63d': 0.20,
        '126d': 0.25,
        '252d': 0.10,
        'rsi': 0.10,
        'macd_hist': 0.10,
        'forca_rel': 0.10
    }
    
    # Normalizar retornos
    for periodo in ['21d', '63d', '126d', '252d']:
        if not np.isnan(metrics.get(periodo, np.nan)):
            normalized = (metrics[periodo] + 50) / 150 * 100
            normalized = max(0, min(100, normalized))
            score += normalized * weights[periodo]
    
    if not np.isnan(metrics.get('rsi', np.nan)):
        score += metrics['rsi'] * weights['rsi']
    
    if not np.isnan(metrics.get('macd_hist', np.nan)):
        normalized = (metrics['macd_hist'] + 2) / 4 * 100
        normalized = max(0, min(100, normalized))
        score += normalized * weights['macd_hist']
    
    if not np.isnan(metrics.get('forca_rel', np.nan)):
        normalized = (metrics['forca_rel'] + 30) / 60 * 100
        normalized = max(0, min(100, normalized))
        score += normalized * weights['forca_rel']
    
    return round(score, 2)


def analisar_acoes(data_dict, ibov_data):
    """Análise completa de todas as ações"""
    resultados = []
    
    for ticker, df in data_dict.items():
        if len(df) < 21:
            continue
        
        try:
            retornos = calcular_retornos(df)
            rsi = calcular_rsi(df)
            macd_line, macd_hist = calcular_macd(df)
            forca_rel_6m = calcular_forca_relativa(df, ibov_data, 126)
            vol = calcular_volatilidade(df)
            
            metrics = {
                **retornos,
                'rsi': rsi,
                'macd': macd_line,
                'macd_hist': macd_hist,
                'forca_rel': forca_rel_6m,
                'volatilidade': vol
            }
            
            momentum_score = calcular_momentum_score(metrics)
            
            # Filtrar outliers absurdos (retornos > 500%)
            if retornos.get('126d', 0) > 500:
                continue
            
            resultados.append({
                'Ticker': ticker.replace('.SA', ''),
                'Setor': ACOES_SETORES.get(ticker, 'Outro'),
                'Score': momentum_score,
                'Ret_1M': retornos.get('21d', np.nan),
                'Ret_3M': retornos.get('63d', np.nan),
                'Ret_6M': retornos.get('126d', np.nan),
                'Ret_12M': retornos.get('252d', np.nan),
                'RSI': rsi,
                'MACD': macd_line,
                'MACD_Hist': macd_hist,
                'Força_Rel_6M': forca_rel_6m,
                'Volatilidade': vol,
                'Preço': df['Close'].iloc[-1]
            })
        except Exception:
            pass
    
    return pd.DataFrame(resultados).sort_values('Score', ascending=False).reset_index(drop=True)


def calcular_score_historico(data_dict, ibov_data, lookback_days=180):
    """Calcula score histórico para gráfico de reversão"""
    scores_historicos = {}
    
    for ticker, df in data_dict.items():
        if len(df) < 126:
            continue
            
        scores = []
        dates = []
        
        # Calcular score a cada 7 dias nos últimos lookback_days
        for i in range(lookback_days, 0, -7):
            if i > len(df):
                continue
                
            df_slice = df.iloc[:-i] if i > 1 else df
            
            if len(df_slice) < 63:
                continue
            
            try:
                retornos = calcular_retornos(df_slice, periodos=[21, 63, min(126, len(df_slice)-1)])
                rsi = calcular_rsi(df_slice)
                macd_line, macd_hist = calcular_macd(df_slice)
                
                # Força relativa simplificada
                forca_rel = 0
                if len(ibov_data) > i and len(df_slice) >= 63:
                    ibov_slice = ibov_data.iloc[:-i] if i > 1 else ibov_data
                    forca_rel = calcular_forca_relativa(df_slice, ibov_slice, min(63, len(df_slice)-1))
                
                metrics = {
                    '21d': retornos.get('21d', 0),
                    '63d': retornos.get('63d', 0),
                    '126d': retornos.get('126d', 0),
                    '252d': 0,
                    'rsi': rsi,
                    'macd_hist': macd_hist,
                    'forca_rel': forca_rel
                }
                
                score = calcular_momentum_score(metrics)
                scores.append(score)
                dates.append(df_slice.index[-1])
            except:
                continue
        
        if len(scores) > 5:
            scores_historicos[ticker] = {
                'dates': dates,
                'scores': scores
            }
    
    return scores_historicos


def criar_gauge_chart(value, title):
    """Cria gráfico gauge para o score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#ffcccc'},
                {'range': [33, 66], 'color': '#ffffcc'},
                {'range': [66, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def criar_heatmap_setores(df):
    """Cria heatmap de momentum por setor"""
    setor_avg = df.groupby('Setor')['Score'].mean().sort_values(ascending=False)
    
    fig = go.Figure(data=go.Bar(
        x=setor_avg.values,
        y=setor_avg.index,
        orientation='h',
        marker=dict(
            color=setor_avg.values,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Score")
        )
    ))
    
    fig.update_layout(
        title="Score Médio de Momentum por Setor",
        xaxis_title="Score Médio",
        yaxis_title="Setor",
        height=500,
        margin=dict(l=200, r=20, t=50, b=20)
    )
    
    return fig


def criar_scatter_melhorado(df):
    """Scatter plot melhorado - mais legível"""
    # Remover outliers extremos de volatilidade para melhor visualização
    df_plot = df[df['Volatilidade'] < 150].copy()
    
    fig = px.scatter(
        df_plot,
        x='Volatilidade',
        y='Ret_6M',
        size='Score',
        color='Setor',
        hover_name='Ticker',
        hover_data={
            'Score': ':.1f',
            'RSI': ':.1f',
            'Volatilidade': ':.1f%',
            'Ret_6M': ':.1f%'
        },
        title='Retorno 6M vs Volatilidade por Setor',
        size_max=25
    )
    
    # Adicionar linhas de referência
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Adicionar quadrantes
    fig.add_annotation(
        x=25, y=df_plot['Ret_6M'].max() * 0.9,
        text="🎯 IDEAL<br>Alto retorno<br>Baixo risco",
        showarrow=False,
        font=dict(size=10, color="green"),
        bgcolor="rgba(144, 238, 144, 0.3)",
        bordercolor="green"
    )
    
    fig.update_layout(
        xaxis_title="Volatilidade Anualizada (%)",
        yaxis_title="Retorno 6 Meses (%)",
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig


# ==================== INTERFACE STREAMLIT ====================

def main():
    st.title("📈 Dashboard de Momentum - Ações B3")
    st.markdown("**Análise multifatorial de momentum para ações brasileiras**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        periodo_analise = st.selectbox(
            "Período de análise",
            ['1y', '2y'],
            index=0,
            help="Período histórico para cálculo dos indicadores"
        )
        
        min_score = st.slider(
            "Score mínimo para exibição",
            0, 100, 0,
            help="Filtrar ações com score acima deste valor"
        )
        
        setores_unicos = sorted(set(ACOES_SETORES.values()))
        setores_selecionados = st.multiselect(
            "Filtrar por setores",
            options=setores_unicos,
            default=setores_unicos
        )
        
        st.markdown("---")
        
        if st.button("🔄 Atualizar Dados", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Sobre o Score:**")
        st.markdown("""
        O Score de Momentum (0-100) combina:
        - ✅ Retornos em múltiplos períodos
        - ✅ RSI (Índice de Força Relativa)
        - ✅ MACD (convergência/divergência)
        - ✅ Força relativa vs Ibovespa
        """)
        
        st.markdown("---")
        st.info(f"💡 Analisando **{len(ACOES_B3)} ações**")
    
    # Download e processamento dos dados
    with st.spinner("⏳ Baixando dados... Isso pode levar 2-3 minutos."):
        data_dict, ibov_data = download_data(ACOES_B3, period=periodo_analise)
    
    if not data_dict:
        st.error("❌ Não foi possível baixar dados. Tente novamente mais tarde.")
        st.stop()
    
    with st.spinner("🔢 Calculando indicadores de momentum..."):
        df_resultado = analisar_acoes(data_dict, ibov_data)
    
    if len(df_resultado) == 0:
        st.error("❌ Não foi possível processar os dados. Tente novamente.")
        st.stop()
    
    # Aplicar filtros
    df_filtrado = df_resultado[
        (df_resultado['Score'] >= min_score) &
        (df_resultado['Setor'].isin(setores_selecionados))
    ]
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Ações", len(df_filtrado))
    
    with col2:
        st.metric("Score Médio", f"{df_filtrado['Score'].mean():.1f}")
    
    with col3:
        if len(df_filtrado) > 0:
            top_score = df_filtrado['Score'].max()
            top_ticker = df_filtrado[df_filtrado['Score'] == top_score]['Ticker'].values[0]
            st.metric("Maior Score", f"{top_ticker} ({top_score:.1f})")
        else:
            st.metric("Maior Score", "N/A")
    
    with col4:
        acima_70 = len(df_filtrado[df_filtrado['Score'] >= 70])
        st.metric("Ações Score > 70", acima_70)
    
    st.markdown("---")
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Ranking", 
        "🎯 Análise Setorial", 
        "📈 Visualizações",
        "🔄 Oportunidades de Reversão",
        "🔍 Detalhes"
    ])
    
    with tab1:
        st.subheader("Ranking de Momentum")
        
        if len(df_filtrado) == 0:
            st.warning("Nenhuma ação encontrada com os filtros selecionados.")
        else:
            col_top, col_bottom = st.columns(2)
            
            with col_top:
                st.markdown("**🔥 Top 10 - Maior Momentum**")
                top10 = df_filtrado.head(min(10, len(df_filtrado)))
                
                for idx, row in top10.iterrows():
                    with st.container():
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        col_a.markdown(f"**{idx+1}. {row['Ticker']}** ({row['Setor']})")
                        col_b.metric("Score", f"{row['Score']:.1f}")
                        col_c.metric("Ret 6M", f"{row['Ret_6M']:.1f}%")
            
            with col_bottom:
                st.markdown("**❄️ Bottom 10 - Menor Momentum**")
                bottom10 = df_filtrado.tail(min(10, len(df_filtrado))).iloc[::-1]
                
                for idx, row in bottom10.iterrows():
                    with st.container():
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        col_a.markdown(f"**{row['Ticker']}** ({row['Setor']})")
                        col_b.metric("Score", f"{row['Score']:.1f}")
                        col_c.metric("Ret 6M", f"{row['Ret_6M']:.1f}%")
            
            st.markdown("---")
            st.markdown("**📋 Tabela Completa**")
            
            df_display = df_filtrado[[
                'Ticker', 'Setor', 'Score', 'Ret_1M', 'Ret_3M', 
                'Ret_6M', 'Ret_12M', 'RSI', 'Força_Rel_6M', 'Volatilidade'
            ]].copy()
            
            def highlight_score(val):
                if pd.isna(val):
                    return ''
                if val >= 70:
                    color = '#90EE90'
                elif val >= 50:
                    color = '#FFFFE0'
                else:
                    color = '#FFB6C1'
                return f'background-color: {color}'
            
            st.dataframe(
                df_display.style.format({
                    'Score': '{:.1f}',
                    'Ret_1M': '{:.1f}%',
                    'Ret_3M': '{:.1f}%',
                    'Ret_6M': '{:.1f}%',
                    'Ret_12M': '{:.1f}%',
                    'RSI': '{:.1f}',
                    'Força_Rel_6M': '{:.1f}%',
                    'Volatilidade': '{:.1f}%'
                }, na_rep='N/A').applymap(highlight_score, subset=['Score']),
                use_container_width=True,
                height=400
            )
    
    with tab2:
        st.subheader("Análise por Setor")
        
        if len(df_filtrado) > 0:
            fig_heatmap = criar_heatmap_setores(df_filtrado)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("**📊 Estatísticas Detalhadas por Setor**")
            
            setor_stats = df_filtrado.groupby('Setor').agg({
                'Score': ['mean', 'max', 'min', 'count'],
                'Ret_6M': 'mean',
                'Volatilidade': 'mean'
            }).round(2)
            
            setor_stats.columns = ['Score Médio', 'Score Max', 'Score Min', 'Qtd Ações', 'Ret 6M Médio', 'Vol Média']
            setor_stats = setor_stats.sort_values('Score Médio', ascending=False)
            
            st.dataframe(setor_stats, use_container_width=True)
        else:
            st.warning("Nenhum dado disponível para análise setorial.")
    
    with tab3:
        st.subheader("Visualizações Avançadas")
        
        if len(df_filtrado) > 0:
            # Scatter melhorado
            fig_scatter = criar_scatter_melhorado(df_filtrado)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                fig_hist = px.histogram(
                    df_filtrado,
                    x='Score',
                    nbins=20,
                    title='Distribuição de Scores',
                    color_discrete_sequence=['#636EFA']
                )
                fig_hist.update_layout(
                    xaxis_title="Score de Momentum",
                    yaxis_title="Quantidade de Ações",
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_dist2:
                fig_box = px.box(
                    df_filtrado,
                    x='Setor',
                    y='Score',
                    title='Distribuição de Score por Setor',
                    color='Setor'
                )
                fig_box.update_layout(
                    xaxis_title="",
                    yaxis_title="Score",
                    showlegend=False,
                    height=400
                )
                fig_box.update_xaxes(tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Nenhum dado disponível para visualizações.")
    
    with tab4:
        st.subheader("🔄 Oportunidades de Reversão")
        
        st.markdown("""
        Ações com **momentum fraco** (score baixo) que podem estar **começando a reverter**.
        
        **Como usar:**
        - Procure ações com score subindo nos últimos meses
        - MACD virando positivo = sinal de reversão
        - RSI saindo de oversold (<30) = pressão vendedora diminuindo
        """)
        
        # Filtrar ações com score baixo
        df_reversao = df_resultado[df_resultado['Score'] < 40].copy()
        df_reversao = df_reversao.sort_values('Score', ascending=False)
        
        if len(df_reversao) == 0:
            st.info("Nenhuma ação com score < 40 no momento.")
        else:
            # Seletor de ação
            ticker_reversao = st.selectbox(
                "Selecione uma ação para ver evolução do score:",
                df_reversao['Ticker'].values,
                key='reversao_select'
            )
            
            ticker_completo = ticker_reversao + '.SA'
            
            # Mostrar métricas atuais
            acao_rev = df_reversao[df_reversao['Ticker'] == ticker_reversao].iloc[0]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Score Atual", f"{acao_rev['Score']:.1f}")
            col2.metric("Ret 6M", f"{acao_rev['Ret_6M']:.1f}%")
            col3.metric("RSI", f"{acao_rev['RSI']:.1f}")
            col4.metric("MACD Hist", f"{acao_rev['MACD_Hist']:.2f}")
            col5.metric("Volatilidade", f"{acao_rev['Volatilidade']:.1f}%")
            
            # Calcular histórico de score
            with st.spinner("Calculando histórico de score..."):
                scores_hist = calcular_score_historico(data_dict, ibov_data, lookback_days=180)
            
            if ticker_completo in scores_hist:
                hist_data = scores_hist[ticker_completo]
                
                # Gráfico de evolução do score
                fig_score_time = go.Figure()
                
                fig_score_time.add_trace(go.Scatter(
                    x=hist_data['dates'],
                    y=hist_data['scores'],
                    mode='lines+markers',
                    name='Score',
                    line=dict(color='#636EFA', width=2),
                    marker=dict(size=6)
                ))
                
                # Linhas de referência
                fig_score_time.add_hline(y=40, line_dash="dash", line_color="orange", 
                                         annotation_text="Limiar Reversão (40)")
                fig_score_time.add_hline(y=70, line_dash="dash", line_color="green",
                                         annotation_text="Momentum Forte (70)")
                
                fig_score_time.update_layout(
                    title=f"Evolução do Score de Momentum - {ticker_reversao}",
                    xaxis_title="Data",
                    yaxis_title="Score",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_score_time, use_container_width=True)
                
                # Análise de tendência
                if len(hist_data['scores']) >= 3:
                    score_recente = hist_data['scores'][-1]
                    score_3m = hist_data['scores'][len(hist_data['scores'])//2]
                    
                    if score_recente > score_3m + 5:
                        st.success(f"✅ **SINAL POSITIVO**: Score subindo! De {score_3m:.1f} → {score_recente:.1f}")
                        st.markdown("Possível reversão em andamento. Monitore se MACD também virou positivo.")
                    elif score_recente < score_3m - 5:
                        st.warning(f"⚠️ **SINAL NEGATIVO**: Score caindo. De {score_3m:.1f} → {score_recente:.1f}")
                        st.markdown("Ainda não há sinais claros de reversão.")
                    else:
                        st.info(f"➡️ **NEUTRO**: Score estável em torno de {score_recente:.1f}")
            else:
                st.warning("Dados históricos insuficientes para esta ação.")
            
            # Tabela de todas as oportunidades
            st.markdown("---")
            st.markdown("**📋 Todas as Oportunidades (Score < 40)**")
            
            df_rev_display = df_reversao[[
                'Ticker', 'Setor', 'Score', 'Ret_1M', 'Ret_6M', 'RSI', 'MACD_Hist'
            ]].copy()
            
            st.dataframe(
                df_rev_display.style.format({
                    'Score': '{:.1f}',
                    'Ret_1M': '{:.1f}%',
                    'Ret_6M': '{:.1f}%',
                    'RSI': '{:.1f}',
                    'MACD_Hist': '{:.2f}'
                }, na_rep='N/A'),
                use_container_width=True
            )
    
    with tab5:
        st.subheader("Análise Detalhada de Ação")
        
        if len(df_filtrado) > 0:
            ticker_selecionado = st.selectbox(
                "Selecione uma ação para análise detalhada",
                df_filtrado['Ticker'].values
            )
            
            acao_data = df_filtrado[df_filtrado['Ticker'] == ticker_selecionado].iloc[0]
            ticker_completo = ticker_selecionado + '.SA'
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("Score", f"{acao_data['Score']:.1f}")
            col2.metric("Ret 6M", f"{acao_data['Ret_6M']:.1f}%")
            col3.metric("RSI", f"{acao_data['RSI']:.1f}")
            col4.metric("Força Rel", f"{acao_data['Força_Rel_6M']:.1f}%")
            col5.metric("Vol", f"{acao_data['Volatilidade']:.1f}%")
            
            st.plotly_chart(
                criar_gauge_chart(acao_data['Score'], f"Score de Momentum - {ticker_selecionado}"),
                use_container_width=True
            )
            
            if ticker_completo in data_dict:
                df_acao = data_dict[ticker_completo].copy()
                
                fig_preco = go.Figure()
                
                fig_preco.add_trace(go.Candlestick(
                    x=df_acao.index,
                    open=df_acao['Open'],
                    high=df_acao['High'],
                    low=df_acao['Low'],
                    close=df_acao['Close'],
                    name='Preço'
                ))
                
                df_acao['MA20'] = df_acao['Close'].rolling(20).mean()
                df_acao['MA50'] = df_acao['Close'].rolling(50).mean()
                
                fig_preco.add_trace(go.Scatter(
                    x=df_acao.index,
                    y=df_acao['MA20'],
                    name='MA20',
                    line=dict(color='orange', width=1)
                ))
                
                fig_preco.add_trace(go.Scatter(
                    x=df_acao.index,
                    y=df_acao['MA50'],
                    name='MA50',
                    line=dict(color='blue', width=1)
                ))
                
                fig_preco.update_layout(
                    title=f"Histórico de Preços - {ticker_selecionado}",
                    yaxis_title="Preço (R$)",
                    xaxis_title="Data",
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig_preco, use_container_width=True)
                
                st.markdown("**📋 Todas as Métricas**")
                
                metricas_detalhadas = pd.DataFrame({
                    'Indicador': [
                        'Retorno 1 Mês', 'Retorno 3 Meses', 'Retorno 6 Meses', 'Retorno 12 Meses',
                        'RSI', 'MACD', 'MACD Histograma', 'Força Relativa 6M', 'Volatilidade',
                        'Preço Atual', 'Setor'
                    ],
                    'Valor': [
                        f"{acao_data['Ret_1M']:.2f}%" if not pd.isna(acao_data['Ret_1M']) else 'N/A',
                        f"{acao_data['Ret_3M']:.2f}%" if not pd.isna(acao_data['Ret_3M']) else 'N/A',
                        f"{acao_data['Ret_6M']:.2f}%" if not pd.isna(acao_data['Ret_6M']) else 'N/A',
                        f"{acao_data['Ret_12M']:.2f}%" if not pd.isna(acao_data['Ret_12M']) else 'N/A',
                        f"{acao_data['RSI']:.2f}" if not pd.isna(acao_data['RSI']) else 'N/A',
                        f"{acao_data['MACD']:.4f}" if not pd.isna(acao_data['MACD']) else 'N/A',
                        f"{acao_data['MACD_Hist']:.4f}" if not pd.isna(acao_data['MACD_Hist']) else 'N/A',
                        f"{acao_data['Força_Rel_6M']:.2f}%" if not pd.isna(acao_data['Força_Rel_6M']) else 'N/A',
                        f"{acao_data['Volatilidade']:.2f}%" if not pd.isna(acao_data['Volatilidade']) else 'N/A',
                        f"R$ {acao_data['Preço']:.2f}" if not pd.isna(acao_data['Preço']) else 'N/A',
                        acao_data['Setor']
                    ]
                })
                
                st.table(metricas_detalhadas)
        else:
            st.warning("Nenhuma ação disponível para análise detalhada.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    📊 Dashboard de Momentum - B3 | {len(df_resultado)} ações analisadas | Atualizado automaticamente
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
