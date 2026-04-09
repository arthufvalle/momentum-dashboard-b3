"""
Dashboard de Índice de Momentum - Ações Brasileiras (B3)
Análise multifatorial de momentum com visualizações interativas
Versão 3.0 - Score histórico 12M + painel de racional
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuração da página
st.set_page_config(
    page_title="Momentum Dashboard - B3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .racional-box {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .racional-title {
        font-weight: 600;
        font-size: 15px;
        margin-bottom: 6px;
        color: #1f77b4;
    }
    .racional-text {
        font-size: 13px;
        color: #444;
        line-height: 1.6;
    }
    .peso-badge {
        display: inline-block;
        background: #1f77b4;
        color: white;
        border-radius: 12px;
        padding: 1px 8px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    </style>
    """, unsafe_allow_html=True)


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


def download_single(ticker, period):
    """Download de um único ticker com retry"""
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, progress=False, timeout=10)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 20:
                return ticker, df
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return ticker, None


@st.cache_data(ttl=7200)
def download_data(tickers, period='2y'):
    """Download paralelo de dados históricos"""
    data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Baixando dados em paralelo...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_single, t, period): t for t in tickers}
        completed = 0
        for future in as_completed(futures):
            ticker, df = future.result()
            if df is not None:
                data[ticker] = df
            completed += 1
            progress_bar.progress(completed / len(tickers))

    progress_bar.empty()
    status_text.empty()

    try:
        ibov = yf.download('^BVSP', period=period, progress=False, timeout=10)
        if isinstance(ibov.columns, pd.MultiIndex):
            ibov.columns = ibov.columns.get_level_values(0)
    except Exception:
        ibov = pd.DataFrame()

    return data, ibov


def calcular_retorno_periodo(df, inicio_dias, fim_dias=0):
    """Retorno entre dois pontos no passado (em dias úteis)"""
    if len(df) < inicio_dias:
        return np.nan
    if fim_dias == 0:
        return (df['Close'].iloc[-1] / df['Close'].iloc[-inicio_dias] - 1) * 100
    else:
        if len(df) < inicio_dias:
            return np.nan
        return (df['Close'].iloc[-fim_dias] / df['Close'].iloc[-inicio_dias] - 1) * 100


def calcular_consistencia(df, meses=12):
    """% de meses com retorno positivo nos últimos N meses"""
    if len(df) < 21 * meses:
        return np.nan
    monthly = df['Close'].resample('ME').last().pct_change().dropna()
    if len(monthly) < meses:
        return np.nan
    ultimos = monthly.tail(meses)
    return (ultimos > 0).sum() / len(ultimos) * 100


def calcular_information_ratio(df_ativo, df_indice, periodo=126):
    """Information ratio: força relativa / volatilidade do excesso de retorno"""
    if len(df_ativo) < periodo or len(df_indice) < periodo:
        return np.nan
    try:
        ret_a = df_ativo['Close'].pct_change().dropna().tail(periodo)
        ret_i = df_indice['Close'].pct_change().dropna().tail(periodo)
        # alinhar índices
        excess = ret_a.values - ret_i.values[:len(ret_a)]
        if excess.std() == 0:
            return np.nan
        ir = (excess.mean() / excess.std()) * np.sqrt(252)
        return ir
    except Exception:
        return np.nan


def calcular_rsi(df, period=14):
    if len(df) < period:
        return np.nan
    try:
        return RSIIndicator(close=df['Close'], window=period).rsi().iloc[-1]
    except Exception:
        return np.nan


def calcular_macd(df):
    if len(df) < 26:
        return np.nan, np.nan
    try:
        macd = MACD(close=df['Close'])
        macd_line = macd.macd().iloc[-1]
        return macd_line, macd_line - macd.macd_signal().iloc[-1]
    except Exception:
        return np.nan, np.nan


def calcular_volatilidade(df, periodo=21):
    if len(df) < periodo:
        return np.nan
    try:
        return df['Close'].pct_change().dropna().tail(periodo).std() * np.sqrt(252) * 100
    except Exception:
        return np.nan


def calcular_momentum_score(metrics):
    """
    Score composto de momentum (0-100) — literatura financeira

    Fatores e pesos:
      50% → Retorno 12M-1M  (Jegadeesh & Titman: horizonte mais preditivo)
      20% → Retorno 6M-1M   (momentum de médio prazo)
      15% → Information Ratio 6M (força relativa ajustada por risco)
      15% → Consistência 12M (% meses positivos — qualidade do momentum)

    Normalização: percentil implícito via clamp em faixas esperadas.
    NaNs redistribuem o peso para os fatores disponíveis.
    """
    fatores = {
        'ret_12m_1m': 0.50,
        'ret_6m_1m':  0.20,
        'ir_6m':      0.15,
        'consist_12m': 0.15,
    }

    def norm_retorno(v, low=-40, high=80):
        if np.isnan(v):
            return np.nan
        return max(0, min(100, (v - low) / (high - low) * 100))

    def norm_ir(v, low=-2, high=2):
        if np.isnan(v):
            return np.nan
        return max(0, min(100, (v - low) / (high - low) * 100))

    def norm_consist(v):
        if np.isnan(v):
            return np.nan
        return max(0, min(100, v))  # já é %

    normalized = {
        'ret_12m_1m':  norm_retorno(metrics.get('ret_12m_1m', np.nan)),
        'ret_6m_1m':   norm_retorno(metrics.get('ret_6m_1m', np.nan), low=-30, high=60),
        'ir_6m':       norm_ir(metrics.get('ir_6m', np.nan)),
        'consist_12m': norm_consist(metrics.get('consist_12m', np.nan)),
    }

    # redistribuir pesos quando fator é NaN
    peso_total = sum(fatores[k] for k, v in normalized.items() if not np.isnan(v))
    if peso_total == 0:
        return np.nan

    score = 0
    for k, v in normalized.items():
        if not np.isnan(v):
            peso_adj = fatores[k] / peso_total
            score += v * peso_adj

    return round(score, 2)


def analisar_acoes(data_dict, ibov_data):
    resultados = []

    for ticker, df in data_dict.items():
        if len(df) < 63:
            continue
        try:
            # Retorno 12M excluindo último mês (Jegadeesh & Titman)
            ret_12m_1m = calcular_retorno_periodo(df, inicio_dias=252, fim_dias=21)
            # Retorno 6M excluindo último mês
            ret_6m_1m  = calcular_retorno_periodo(df, inicio_dias=126, fim_dias=21)
            # Retornos brutos para exibição
            ret_1m  = calcular_retorno_periodo(df, 21)
            ret_3m  = calcular_retorno_periodo(df, 63)
            ret_6m  = calcular_retorno_periodo(df, 126)
            ret_12m = calcular_retorno_periodo(df, 252)

            ir_6m       = calcular_information_ratio(df, ibov_data, 126)
            consist_12m = calcular_consistencia(df, 12)
            rsi         = calcular_rsi(df)
            macd_line, macd_hist = calcular_macd(df)
            vol         = calcular_volatilidade(df)

            metrics = {
                'ret_12m_1m':  ret_12m_1m,
                'ret_6m_1m':   ret_6m_1m,
                'ir_6m':       ir_6m,
                'consist_12m': consist_12m,
            }

            score = calcular_momentum_score(metrics)
            if score is None or np.isnan(score):
                continue
            if ret_6m is not None and not np.isnan(ret_6m) and ret_6m > 500:
                continue

            resultados.append({
                'Ticker':       ticker.replace('.SA', ''),
                'Setor':        ACOES_SETORES.get(ticker, 'Outro'),
                'Score':        score,
                'Ret_1M':       ret_1m,
                'Ret_3M':       ret_3m,
                'Ret_6M':       ret_6m,
                'Ret_12M':      ret_12m,
                'Ret_12M_1M':   ret_12m_1m,
                'Ret_6M_1M':    ret_6m_1m,
                'IR_6M':        ir_6m,
                'Consist_12M':  consist_12m,
                'RSI':          rsi,
                'MACD':         macd_line,
                'MACD_Hist':    macd_hist,
                'Volatilidade': vol,
                'Preço':        df['Close'].iloc[-1],
            })
        except Exception:
            pass

    return pd.DataFrame(resultados).sort_values('Score', ascending=False).reset_index(drop=True)


@st.cache_data(ttl=7200)
def calcular_score_historico_cached(ticker, _df, _ibov, n_pontos=52):
    """
    Calcula série semanal do score ao longo de 1 ano.
    Cacheado por ticker — não recalcula ao mudar seleção.
    """
    scores = []
    dates  = []
    step   = 5  # dias úteis por ponto (~1 semana)

    total_steps = n_pontos
    for i in range(total_steps * step, 0, -step):
        if i >= len(_df):
            continue
        df_slice   = _df.iloc[:-i]
        ibov_slice = _ibov.iloc[:-i] if i < len(_ibov) else _ibov

        if len(df_slice) < 63:
            continue
        try:
            ret_12m_1m  = calcular_retorno_periodo(df_slice, 252, 21)
            ret_6m_1m   = calcular_retorno_periodo(df_slice, 126, 21)
            ir_6m       = calcular_information_ratio(df_slice, ibov_slice, 126)
            consist_12m = calcular_consistencia(df_slice, 12)

            score = calcular_momentum_score({
                'ret_12m_1m':  ret_12m_1m,
                'ret_6m_1m':   ret_6m_1m,
                'ir_6m':       ir_6m,
                'consist_12m': consist_12m,
            })
            if score is not None and not np.isnan(score):
                scores.append(score)
                dates.append(df_slice.index[-1])
        except Exception:
            continue

    return dates, scores


def criar_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33],  'color': '#ffcccc'},
                {'range': [33, 66], 'color': '#ffffcc'},
                {'range': [66, 100],'color': '#ccffcc'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def criar_heatmap_setores(df):
    setor_avg = df.groupby('Setor')['Score'].mean().sort_values(ascending=False)
    fig = go.Figure(data=go.Bar(
        x=setor_avg.values, y=setor_avg.index, orientation='h',
        marker=dict(color=setor_avg.values, colorscale='RdYlGn', showscale=True)
    ))
    fig.update_layout(
        title="Score médio de momentum por setor",
        xaxis_title="Score médio", height=500,
        margin=dict(l=200, r=20, t=50, b=20)
    )
    return fig


def criar_scatter(df):
    df_plot = df[df['Volatilidade'] < 150].copy()
    fig = px.scatter(
        df_plot, x='Volatilidade', y='Ret_6M',
        size='Score', color='Setor', hover_name='Ticker',
        hover_data={'Score': ':.1f', 'RSI': ':.1f', 'Volatilidade': ':.1f', 'Ret_6M': ':.1f'},
        title='Retorno 6M vs volatilidade por setor', size_max=25
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(xaxis_title="Volatilidade anualizada (%)", yaxis_title="Retorno 6M (%)", height=600)
    return fig


# ============================================================
#  PAINEL DE RACIONAL
# ============================================================

def render_racional():
    st.subheader("📖 Como o score é calculado")

    st.markdown("""
    O **Score de Momentum** é um índice composto de 0 a 100 que combina quatro fatores derivados
    da literatura acadêmica de fator de momentum (Jegadeesh & Titman, 1993; Asness et al., 2013).
    Quanto maior o score, mais forte e consistente é o momentum da ação.
    """)

    st.markdown("---")
    st.markdown("### Fatores e pesos")

    fatores = [
        {
            "nome": "Retorno 12M excluindo o último mês",
            "peso": "50%",
            "cor": "#1f77b4",
            "texto": """
            O fator mais importante do modelo. Mede o retorno acumulado nos últimos 12 meses,
            <strong>excluindo deliberadamente o mês mais recente</strong>. Esse ajuste existe porque há
            um efeito bem documentado de <em>reversão de curto prazo</em>: ações que subiram muito
            no último mês tendem a corrigir na média nas semanas seguintes. Ao excluir o 1M,
            o fator captura tendência estrutural e reduz ruído de curto prazo.
            <br><br>
            <strong>No contexto brasileiro:</strong> o efeito de reversão de 1M é ainda mais pronunciado
            que em mercados desenvolvidos, tornando essa exclusão especialmente relevante.
            """
        },
        {
            "nome": "Retorno 6M excluindo o último mês",
            "peso": "20%",
            "cor": "#1f77b4",
            "texto": """
            Complementa o fator de 12M com um horizonte mais curto. Captura ações que estão
            em fase de aceleração de momentum — não precisaram de 12 meses para se destacar,
            mas mostram força relativa clara nos últimos 6 meses. Também exclui o último mês
            pelo mesmo racional acima.
            """
        },
        {
            "nome": "Information Ratio 6M",
            "peso": "15%",
            "cor": "#2ca02c",
            "texto": """
            Mede a <strong>força relativa ajustada por risco</strong> vs. Ibovespa no período de 6 meses.
            É calculado como:<br><br>
            <code>IR = média(retorno diário da ação − retorno diário do Ibov) / desvio padrão do excesso × √252</code>
            <br><br>
            Um IR positivo significa que a ação consistentemente gerou retorno acima do índice,
            não só em dias específicos. É mais robusto que simplesmente comparar retornos acumulados
            porque penaliza volatilidade excessiva. Um IR de 0.5 ou mais é considerado bom.
            """
        },
        {
            "nome": "Consistência 12M",
            "peso": "15%",
            "cor": "#ff7f0e",
            "texto": """
            Percentual de meses com retorno positivo nos últimos 12 meses. Diferencia ações com
            <strong>momentum de qualidade</strong> (sobe consistentemente, mês a mês) de ações com
            momentum concentrado em poucos eventos (subiu 80% num mês e ficou parada no resto).
            <br><br>
            Uma ação com 9 de 12 meses positivos (75%) é muito mais interessante do ponto de
            vista de timing do que uma que teve um único spike. Este fator captura a
            <em>persistência</em> da tendência.
            """
        },
    ]

    for f in fatores:
        st.markdown(f"""
        <div class="racional-box">
            <div class="racional-title">
                <span class="peso-badge">{f['peso']}</span>{f['nome']}
            </div>
            <div class="racional-text">{f['texto']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Como interpretar o score")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("**Score 0–33 → Momentum fraco**\nAção sem tendência clara ou em queda. Candidata a short no long/short ou a evitar no long only.")
    with col2:
        st.warning("**Score 33–66 → Momentum neutro**\nSinais mistos. Pode estar em transição — acompanhar de perto para identificar virada.")
    with col3:
        st.success("**Score 66–100 → Momentum forte**\nTendência clara, consistente e acima do Ibov. Zona de interesse para long.")

    st.markdown("---")
    st.markdown("### O que o score NÃO captura")
    st.info("""
    O score é **puramente baseado em preço**. Ele não sabe nada sobre valuation, fundamentos,
    qualidade do balanço ou macro. Uma ação cara pode ter score alto; uma barata pode ter score baixo.

    O uso correto é como **filtro de timing e confirmação de tese**: se você já tem uma tese
    fundamentalista em uma ação, o score ajuda a decidir se o mercado já está concordando
    (momentum alto = narrativa virando) ou se você está early demais (momentum baixo = mercado
    ainda não viu o que você viu).
    """)

    st.markdown("### Limitações conhecidas")
    st.markdown("""
    - **Liquidez não é considerada**: ações ilíquidas podem ter scores altos por movimentos de
      preço artificiais — sempre checar volume antes de agir.
    - **Lookback fixo**: o modelo não se adapta a regimes de mercado (bull vs. bear). Em mercados
      em queda generalizada, scores altos podem ser menos discriminantes.
    - **Dados via Yahoo Finance**: pode haver ajustes de dividendos e splits com defasagem.
      Para uso em produção, recomenda-se Bloomberg como fonte de preços.
    """)


# ============================================================
#  MAIN
# ============================================================

def main():
    st.title("📈 Dashboard de Momentum - Ações B3")
    st.markdown("**Análise multifatorial de momentum para ações brasileiras**")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configurações")

        periodo_analise = st.selectbox(
            "Período de análise",
            ['2y', '3y'],
            index=0,
            help="Período histórico. Mínimo 2 anos para cálculo completo do score."
        )

        min_score = st.slider("Score mínimo", 0, 100, 0)

        setores_unicos = sorted(set(ACOES_SETORES.values()))
        setores_selecionados = st.multiselect(
            "Filtrar por setores",
            options=setores_unicos,
            default=setores_unicos
        )

        st.markdown("---")
        if st.button("🔄 Atualizar dados", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.info(f"📊 Analisando **{len(ACOES_B3)} ações**")
        st.markdown("""
        **Score = média ponderada de:**
        - 50% Retorno 12M-1M
        - 20% Retorno 6M-1M
        - 15% Information Ratio 6M
        - 15% Consistência mensal 12M
        """)

    with st.spinner("⏳ Baixando dados em paralelo..."):
        data_dict, ibov_data = download_data(tuple(ACOES_B3), period=periodo_analise)

    if not data_dict:
        st.error("❌ Não foi possível baixar dados.")
        st.stop()

    with st.spinner("🔢 Calculando scores..."):
        df_resultado = analisar_acoes(data_dict, ibov_data)

    if len(df_resultado) == 0:
        st.error("❌ Não foi possível processar os dados.")
        st.stop()

    df_filtrado = df_resultado[
        (df_resultado['Score'] >= min_score) &
        (df_resultado['Setor'].isin(setores_selecionados))
    ]

    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de ações", len(df_filtrado))
    with col2:
        st.metric("Score médio", f"{df_filtrado['Score'].mean():.1f}")
    with col3:
        if len(df_filtrado) > 0:
            row = df_filtrado.iloc[0]
            st.metric("Maior score", f"{row['Ticker']} ({row['Score']:.1f})")
    with col4:
        st.metric("Ações score > 70", len(df_filtrado[df_filtrado['Score'] >= 70]))

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Ranking",
        "📈 Score no Tempo",
        "🎯 Análise Setorial",
        "📉 Visualizações",
        "🔄 Reversão",
        "📖 Racional"
    ])

    # ── TAB 1: RANKING ──────────────────────────────────────
    with tab1:
        st.subheader("Ranking de momentum")

        if len(df_filtrado) == 0:
            st.warning("Nenhuma ação com os filtros selecionados.")
        else:
            col_top, col_bottom = st.columns(2)

            with col_top:
                st.markdown("**🔥 Top 10 — maior momentum**")
                for i, (_, row) in enumerate(df_filtrado.head(10).iterrows()):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    c1.markdown(f"**{i+1}. {row['Ticker']}** ({row['Setor']})")
                    c2.metric("Score", f"{row['Score']:.1f}")
                    c3.metric("Ret 6M", f"{row['Ret_6M']:.1f}%" if not np.isnan(row['Ret_6M']) else "N/A")

            with col_bottom:
                st.markdown("**❄️ Bottom 10 — menor momentum**")
                for _, row in df_filtrado.tail(10).iloc[::-1].iterrows():
                    c1, c2, c3 = st.columns([2, 1, 1])
                    c1.markdown(f"**{row['Ticker']}** ({row['Setor']})")
                    c2.metric("Score", f"{row['Score']:.1f}")
                    c3.metric("Ret 6M", f"{row['Ret_6M']:.1f}%" if not np.isnan(row['Ret_6M']) else "N/A")

            st.markdown("---")
            st.markdown("**📋 Tabela completa**")

            df_display = df_filtrado[[
                'Ticker', 'Setor', 'Score', 'Ret_1M', 'Ret_3M',
                'Ret_6M', 'Ret_12M', 'IR_6M', 'Consist_12M', 'Volatilidade'
            ]].copy()

            def highlight_score(val):
                if pd.isna(val):
                    return ''
                if val >= 70:   return 'background-color: #90EE90'
                if val >= 50:   return 'background-color: #FFFFE0'
                return 'background-color: #FFB6C1'

            st.dataframe(
                df_display.style.format({
                    'Score': '{:.1f}', 'Ret_1M': '{:.1f}%', 'Ret_3M': '{:.1f}%',
                    'Ret_6M': '{:.1f}%', 'Ret_12M': '{:.1f}%',
                    'IR_6M': '{:.2f}', 'Consist_12M': '{:.1f}%', 'Volatilidade': '{:.1f}%'
                }, na_rep='N/A').map(highlight_score, subset=['Score']),
                use_container_width=True, height=400
            )

    # ── TAB 2: SCORE NO TEMPO ────────────────────────────────
    with tab2:
        st.subheader("📈 Evolução do score — últimos 12 meses")

        st.markdown("""
        Acompanhe como o score de momentum de cada ação evoluiu semana a semana
        ao longo do último ano. Isso permite identificar:
        - Ações **acelerando** (score subindo) → momentum em construção
        - Ações **desacelerando** (score caindo) → possível ponto de saída
        - Ações **revertendo** de score baixo → candidatas a long antecipado
        """)

        if len(df_filtrado) == 0:
            st.warning("Nenhuma ação disponível.")
        else:
            col_sel1, col_sel2 = st.columns([2, 1])

            with col_sel1:
                # Permitir múltiplas ações para comparação
                tickers_disponiveis = df_filtrado['Ticker'].tolist()
                top5_default = tickers_disponiveis[:min(5, len(tickers_disponiveis))]
                tickers_escolhidos = st.multiselect(
                    "Selecione ações para comparar (até 10):",
                    options=tickers_disponiveis,
                    default=top5_default,
                    max_selections=10
                )

            with col_sel2:
                mostrar_ibov_linha = st.checkbox("Mostrar linha de referência (score 50)", value=True)

            if not tickers_escolhidos:
                st.info("Selecione pelo menos uma ação.")
            else:
                fig_tempo = go.Figure()

                cores = px.colors.qualitative.Plotly

                for i, ticker in enumerate(tickers_escolhidos):
                    ticker_sa = ticker + '.SA'
                    if ticker_sa not in data_dict:
                        continue

                    with st.spinner(f"Calculando histórico de {ticker}..."):
                        dates, scores = calcular_score_historico_cached(
                            ticker_sa,
                            data_dict[ticker_sa],
                            ibov_data,
                            n_pontos=52
                        )

                    if len(scores) < 3:
                        continue

                    score_atual = df_filtrado[df_filtrado['Ticker'] == ticker]['Score'].values
                    score_label = f"{score_atual[0]:.1f}" if len(score_atual) > 0 else ""

                    fig_tempo.add_trace(go.Scatter(
                        x=dates, y=scores,
                        mode='lines+markers',
                        name=f"{ticker} (atual: {score_label})",
                        line=dict(color=cores[i % len(cores)], width=2),
                        marker=dict(size=4)
                    ))

                if mostrar_ibov_linha:
                    fig_tempo.add_hline(
                        y=50, line_dash="dot", line_color="gray",
                        annotation_text="Neutro (50)", annotation_position="bottom right"
                    )
                fig_tempo.add_hline(
                    y=70, line_dash="dash", line_color="green",
                    annotation_text="Momentum forte (70)", annotation_position="bottom right",
                    opacity=0.5
                )
                fig_tempo.add_hline(
                    y=33, line_dash="dash", line_color="red",
                    annotation_text="Momentum fraco (33)", annotation_position="bottom right",
                    opacity=0.5
                )

                fig_tempo.update_layout(
                    title="Evolução semanal do score de momentum — últimos 12 meses",
                    xaxis_title="Data",
                    yaxis_title="Score (0–100)",
                    yaxis=dict(range=[0, 100]),
                    height=550,
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )

                st.plotly_chart(fig_tempo, use_container_width=True)

                # Tabela resumo da evolução
                st.markdown("**Variação do score no período**")
                resumo = []
                for ticker in tickers_escolhidos:
                    ticker_sa = ticker + '.SA'
                    if ticker_sa not in data_dict:
                        continue
                    dates, scores = calcular_score_historico_cached(
                        ticker_sa, data_dict[ticker_sa], ibov_data, n_pontos=52
                    )
                    if len(scores) >= 2:
                        variacao = scores[-1] - scores[0]
                        score_min = min(scores)
                        score_max = max(scores)
                        tendencia = "↑ Acelerando" if variacao > 5 else ("↓ Desacelerando" if variacao < -5 else "→ Estável")
                        resumo.append({
                            'Ticker': ticker,
                            'Score atual': round(scores[-1], 1),
                            'Score 12M atrás': round(scores[0], 1),
                            'Variação': round(variacao, 1),
                            'Mínimo': round(score_min, 1),
                            'Máximo': round(score_max, 1),
                            'Tendência': tendencia,
                        })

                if resumo:
                    df_resumo = pd.DataFrame(resumo)
                    st.dataframe(df_resumo, use_container_width=True, hide_index=True)

    # ── TAB 3: SETORIAL ─────────────────────────────────────
    with tab3:
        st.subheader("Análise por setor")
        if len(df_filtrado) > 0:
            st.plotly_chart(criar_heatmap_setores(df_filtrado), use_container_width=True)
            setor_stats = df_filtrado.groupby('Setor').agg({
                'Score': ['mean', 'max', 'min', 'count'],
                'Ret_6M': 'mean', 'Volatilidade': 'mean'
            }).round(2)
            setor_stats.columns = ['Score médio', 'Score max', 'Score min', 'Qtd', 'Ret 6M médio', 'Vol média']
            st.dataframe(setor_stats.sort_values('Score médio', ascending=False), use_container_width=True)

    # ── TAB 4: VISUALIZAÇÕES ─────────────────────────────────
    with tab4:
        st.subheader("Visualizações avançadas")
        if len(df_filtrado) > 0:
            st.plotly_chart(criar_scatter(df_filtrado), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(df_filtrado, x='Score', nbins=20,
                    title='Distribuição de scores', color_discrete_sequence=['#636EFA'])
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                fig_box = px.box(df_filtrado, x='Setor', y='Score',
                    title='Score por setor', color='Setor')
                fig_box.update_layout(height=400, showlegend=False)
                fig_box.update_xaxes(tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)

    # ── TAB 5: REVERSÃO ──────────────────────────────────────
    with tab5:
        st.subheader("🔄 Oportunidades de reversão")
        st.markdown("""
        Ações com **momentum fraco (score < 40)** monitoradas para sinais de virada.
        Use o gráfico de evolução temporal para confirmar se o score está subindo.
        """)

        df_rev = df_resultado[df_resultado['Score'] < 40].sort_values('Score', ascending=False)

        if len(df_rev) == 0:
            st.info("Nenhuma ação com score < 40 no momento.")
        else:
            ticker_rev = st.selectbox("Ação para análise:", df_rev['Ticker'].values)
            row_rev = df_rev[df_rev['Ticker'] == ticker_rev].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score atual", f"{row_rev['Score']:.1f}")
            c2.metric("Ret 6M", f"{row_rev['Ret_6M']:.1f}%" if not np.isnan(row_rev['Ret_6M']) else "N/A")
            c3.metric("IR 6M", f"{row_rev['IR_6M']:.2f}" if not np.isnan(row_rev['IR_6M']) else "N/A")
            c4.metric("Consist. 12M", f"{row_rev['Consist_12M']:.1f}%" if not np.isnan(row_rev['Consist_12M']) else "N/A")

            ticker_sa = ticker_rev + '.SA'
            if ticker_sa in data_dict:
                with st.spinner("Calculando histórico..."):
                    dates, scores = calcular_score_historico_cached(
                        ticker_sa, data_dict[ticker_sa], ibov_data, n_pontos=52
                    )

                if len(scores) >= 3:
                    fig_rev = go.Figure()
                    fig_rev.add_trace(go.Scatter(
                        x=dates, y=scores, mode='lines+markers',
                        line=dict(color='#636EFA', width=2), marker=dict(size=5)
                    ))
                    fig_rev.add_hline(y=40, line_dash="dash", line_color="orange",
                                      annotation_text="Limiar reversão (40)")
                    fig_rev.add_hline(y=70, line_dash="dash", line_color="green",
                                      annotation_text="Momentum forte (70)")
                    fig_rev.update_layout(
                        title=f"Evolução do score — {ticker_rev}",
                        yaxis=dict(range=[0, 100]), height=450, hovermode='x unified'
                    )
                    st.plotly_chart(fig_rev, use_container_width=True)

                    variacao = scores[-1] - scores[0]
                    if variacao > 5:
                        st.success(f"✅ Score subindo +{variacao:.1f} pts no período — possível reversão em andamento.")
                    elif variacao < -5:
                        st.warning(f"⚠️ Score caindo {variacao:.1f} pts — sem sinal de reversão ainda.")
                    else:
                        st.info(f"➡️ Score estável. Aguardar catalisador.")

            st.markdown("---")
            st.markdown("**Todas as candidatas (score < 40)**")
            st.dataframe(
                df_rev[['Ticker', 'Setor', 'Score', 'Ret_1M', 'Ret_6M', 'IR_6M', 'Consist_12M']].style.format({
                    'Score': '{:.1f}', 'Ret_1M': '{:.1f}%', 'Ret_6M': '{:.1f}%',
                    'IR_6M': '{:.2f}', 'Consist_12M': '{:.1f}%'
                }, na_rep='N/A'),
                use_container_width=True
            )

    # ── TAB 6: RACIONAL ──────────────────────────────────────
    with tab6:
        render_racional()

    st.markdown("---")
    st.markdown(f"""
    <div style='text-align:center;color:gray;font-size:12px;'>
    📊 Momentum Dashboard B3 v3.0 | {len(df_resultado)} ações analisadas |
    Score: 50% Ret12M-1M · 20% Ret6M-1M · 15% IR6M · 15% Consistência
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
