"""
Dashboard de Índice de Momentum - Ações Brasileiras (B3)
Análise multifatorial de momentum com visualizações interativas
Versão otimizada para Streamlit Cloud
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


# Lista de ações brasileiras mais líquidas
ACOES_B3 = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
    'BBAS3.SA', 'WEGE3.SA', 'RENT3.SA', 'SUZB3.SA', 'RAIL3.SA',
    'MGLU3.SA', 'LREN3.SA', 'HAPV3.SA', 'EMBR3.SA', 'ELET3.SA',
    'RADL3.SA', 'JBSS3.SA', 'BEEF3.SA', 'PRIO3.SA', 'GGBR4.SA',
    'CSNA3.SA', 'USIM5.SA', 'CSAN3.SA', 'VIVT3.SA', 'KLBN11.SA',
    'ECOR3.SA', 'EQTL3.SA', 'SANB11.SA', 'CYRE3.SA', 'MRVE3.SA',
    'TOTS3.SA', 'AZZA3.SA', 'CIEL3.SA', 'PCAR3.SA', 'BRFS3.SA',
    'GOAU4.SA', 'MULT3.SA', 'AZUL4.SA', 'GOLL4.SA', 'CCRO3.SA'
]

SETORES = {
    'PETR4.SA': 'Petróleo', 'VALE3.SA': 'Mineração', 'ITUB4.SA': 'Bancos',
    'BBDC4.SA': 'Bancos', 'ABEV3.SA': 'Bebidas', 'BBAS3.SA': 'Bancos',
    'WEGE3.SA': 'Industrial', 'RENT3.SA': 'Locação', 'SUZB3.SA': 'Papel',
    'RAIL3.SA': 'Energia', 'MGLU3.SA': 'Varejo', 'LREN3.SA': 'Varejo',
    'HAPV3.SA': 'Saúde', 'EMBR3.SA': 'Aviação', 'ELET3.SA': 'Energia',
    'RADL3.SA': 'Farmácia', 'JBSS3.SA': 'Frigorífico', 'BEEF3.SA': 'Frigorífico',
    'PRIO3.SA': 'Petróleo', 'GGBR4.SA': 'Siderurgia', 'CSNA3.SA': 'Siderurgia',
    'USIM5.SA': 'Siderurgia', 'CSAN3.SA': 'Energia', 'VIVT3.SA': 'Telecom',
    'KLBN11.SA': 'Papel', 'ECOR3.SA': 'Construção', 'EQTL3.SA': 'Energia',
    'SANB11.SA': 'Bancos', 'CYRE3.SA': 'Construção', 'MRVE3.SA': 'Construção',
    'TOTS3.SA': 'Educação', 'AZZA3.SA': 'Varejo', 'CIEL3.SA': 'Pagamentos',
    'PCAR3.SA': 'Aluguel de Carros', 'BRFS3.SA': 'Frigorífico', 'GOAU4.SA': 'Siderurgia',
    'MULT3.SA': 'Varejo', 'AZUL4.SA': 'Aviação', 'GOLL4.SA': 'Aviação',
    'CCRO3.SA': 'Rodovias'
}


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
                
                if not df.empty and len(df) > 20:  # Mínimo de dados
                    data[ticker] = df
                    break
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(1)
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    st.warning(f"⚠️ Não foi possível baixar {ticker}")
                else:
                    time.sleep(1)
        
        progress_bar.progress((idx + 1) / len(tickers))
    
    progress_bar.empty()
    status_text.empty()
    
    # Download do índice de referência
    try:
        ibov = yf.download('^BVSP', period=period, progress=False, timeout=10)
    except:
        st.error("Erro ao baixar dados do Ibovespa")
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
    """
    Calcula score composto de momentum (0-100)
    """
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
    
    # RSI já está em 0-100
    if not np.isnan(metrics.get('rsi', np.nan)):
        score += metrics['rsi'] * weights['rsi']
    
    # MACD histogram
    if not np.isnan(metrics.get('macd_hist', np.nan)):
        normalized = (metrics['macd_hist'] + 2) / 4 * 100
        normalized = max(0, min(100, normalized))
        score += normalized * weights['macd_hist']
    
    # Força relativa
    if not np.isnan(metrics.get('forca_rel', np.nan)):
        normalized = (metrics['forca_rel'] + 30) / 60 * 100
        normalized = max(0, min(100, normalized))
        score += normalized * weights['forca_rel']
    
    return round(score, 2)


def analisar_acoes(data_dict, ibov_data):
    """Análise completa de todas as ações"""
    resultados = []
    
    for ticker, df in data_dict.items():
        if len(df) == 0 or len(df) < 21:
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
            
            resultados.append({
                'Ticker': ticker.replace('.SA', ''),
                'Setor': SETORES.get(ticker, 'Outro'),
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
        except Exception as e:
            st.warning(f"Erro ao processar {ticker}: {str(e)}")
    
    return pd.DataFrame(resultados).sort_values('Score', ascending=False).reset_index(drop=True)


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
        height=400,
        margin=dict(l=150, r=20, t=50, b=20)
    )
    
    return fig


def criar_scatter_risco_retorno(df):
    """Scatter plot de retorno vs volatilidade"""
    fig = px.scatter(
        df,
        x='Volatilidade',
        y='Ret_6M',
        size='Score',
        color='Score',
        hover_name='Ticker',
        hover_data=['Setor', 'RSI', 'Força_Rel_6M'],
        color_continuous_scale='RdYlGn',
        title='Retorno 6M vs Volatilidade (tamanho = Score Momentum)'
    )
    
    fig.update_layout(
        xaxis_title="Volatilidade Anualizada (%)",
        yaxis_title="Retorno 6 Meses (%)",
        height=500
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
        
        setores_selecionados = st.multiselect(
            "Filtrar por setores",
            options=sorted(set(SETORES.values())),
            default=sorted(set(SETORES.values()))
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
        st.info("💡 **Dica:** Dados são atualizados a cada 2 horas automaticamente")
    
    # Download e processamento dos dados
    with st.spinner("⏳ Baixando dados do mercado... Isso pode levar 1-2 minutos."):
        data_dict, ibov_data = download_data(ACOES_B3, period=periodo_analise)
    
    if not data_dict:
        st.error("❌ Não foi possível baixar dados. Tente novamente mais tarde.")
        st.stop()
    
    if len(ibov_data) == 0:
        st.warning("⚠️ Dados do Ibovespa indisponíveis. Alguns cálculos podem estar limitados.")
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ranking", "🎯 Análise Setorial", "📈 Visualizações", "🔍 Detalhes"])
    
    with tab1:
        st.subheader("Ranking de Momentum")
        
        if len(df_filtrado) == 0:
            st.warning("Nenhuma ação encontrada com os filtros selecionados.")
        else:
            # Top 10
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
            
            # Tabela completa
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
            fig_scatter = criar_scatter_risco_retorno(df_filtrado)
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
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    📊 Dashboard de Momentum - B3 | Dados via yfinance | Atualizado automaticamente a cada 2 horas
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
