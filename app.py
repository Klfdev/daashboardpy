# app.py — Dashboard de Vendas Profissional (Modelo B)
# 
# VERSÃO FINAL: Funcionalidade ChatGPT removida.
# 

import os
import io
import tempfile
import json
from datetime import datetime
import textwrap

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF # Assume-se fpdf2 instalado

# analytics / anomaly
from sklearn.preprocessing import StandardScaler

# IMPORT E VARIAVEIS DA OPENAI REMOVIDOS PARA PUBLICAÇÃO
# As funcionalidades restantes de análise e PDF foram mantidas.


# --------- CONFIG GLOBAL ----------
st.set_page_config(page_title="Dashboard Profissional de Vendas",
                   page_icon="💼",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------
# 🌟 GARANTIA DE INICIALIZAÇÃO DO ESTADO 🌟
# ----------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'orig_columns' not in st.session_state:
    st.session_state.orig_columns = []
# ----------------------------------------------------

# ----------------------------------------------------
# 🌟 CLASSE FPDF CUSTOMIZADA PARA LAYOUT AVANÇADO 🌟
# ----------------------------------------------------
class PDF(FPDF):
    def __init__(self, orientation='P', unit='mm', format='A4', primary_color="#58a6ff"):
        super().__init__(orientation, unit, format)
        self.primary_color = primary_color # Cor primária do dashboard
        self.set_auto_page_break(auto=True, margin=15)

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def header(self):
        r, g, b = self.hex_to_rgb(self.primary_color)
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255) # Branco
        
        self.set_font('Arial', 'B', 14)
        page_width = self.w - 2 * self.l_margin
        
        # Título do Cabeçalho
        self.cell(page_width, 8, self.title, 0, 1, 'L', 1)
        
        # Nova linha e volta às cores normais
        self.ln(5)
        self.set_text_color(0, 0, 0)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(100, 100, 100) # Cinza
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')
        
    def section_title(self, title):
        r, g, b = self.hex_to_rgb(self.primary_color)
        self.set_text_color(r, g, b)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.set_line_width(0.5)
        self.set_draw_color(r, g, b)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)
        self.set_text_color(0, 0, 0) # Volta ao preto para o corpo
        self.set_font('Arial', '', 11)

# ----------------------------------------------------
# FIM DA CLASSE FPDF CUSTOMIZADA
# ----------------------------------------------------

# --------- CSS / THEME ----------
def apply_css(theme="Dark", primary="#58a6ff"):
    # 🌟 CORREÇÃO DE NameError: Inicializa variáveis de cor 🌟
    bg = "#0d1117"    
    text = "#e6e6e6"  
    card = "#121418"  

    if theme == "Dark":
        bg = "#0d1117"
        text = "#e6e6e6"
        card = "#121418"
    elif theme == "Light":
        bg = "#ffffff"
        text = "#111111"
        card = "#f6f8fa"

    css = f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    .metric-card {{
        background: {card};
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 8px;
    }}
    .neon-title {{ color: {primary}; text-shadow: 0 0 8px {primary}22 }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Sidebar quick prefs
st.sidebar.title("⚙️ Configurações")
theme_choice = st.sidebar.radio("Tema", ["Dark", "Light"])
primary_color = st.sidebar.color_picker("Cor primária", "#58a6ff")
apply_css(theme_choice, primary_color)

st.sidebar.markdown("---")
st.sidebar.header("📤 Upload - Planilha (Modelo B)")

# file upload + button workflow
uploaded_file = st.sidebar.file_uploader("Selecione CSV / XLSX", type=["csv", "xlsx", "xls"])
send_button = st.sidebar.button("📨 Enviar Planilha")

st.sidebar.markdown("Formato esperado (Modelo B):")
st.sidebar.markdown("""
`Data, ID Venda, Cliente, Cidade, Estado, Produto, Categoria, Vendedor, Quantidade, Preço Unitário, Desconto, Custo, Total, Lucro`
""")

# sample download
if st.sidebar.button("📥 Baixar planilha exemplo (Modelo B)"):
    sample = pd.DataFrame([
        {"Data":"2025-01-05","ID Venda":"V0001","Cliente":"Cliente A","Cidade":"SP","Estado":"SP","Produto":"Racao Premium","Categoria":"Ração","Vendedor":"João","Quantidade":2,"Preço Unitário":89.9,"Desconto":0,"Custo":60,"Total":179.8,"Lucro":59.8},
        {"Data":"2025-01-07","ID Venda":"V0002","Cliente":"Cliente B","Cidade":"BH","Estado":"MG","Produto":"Brinquedo Bola","Categoria":"Brinquedos","Vendedor":"Maria","Quantidade":1,"Preço Unitário":19.9,"Desconto":0,"Custo":5,"Total":19.9,"Lucro":14.9},
        {"Data":"2025-02-02","ID Venda":"V0003","Cliente":"Cliente C","Cidade":"RJ","Estado":"RJ","Produto":"Shampoo Pet","Categoria":"Higiene","Vendedor":"Ana","Quantidade":3,"Preço Unitário":39.9,"Desconto":5,"Custo":10,"Total":109.7,"Lucro":79.7},
    ])
    st.sidebar.download_button("Download exemplo (CSV)", data=sample.to_csv(index=False).encode("utf-8"),
                               file_name="vendas_exemplo_modeloB.csv", mime="text/csv")

# -------- helper: intelligent import for sales (Model B) ----------
def import_sales(uploaded, dayfirst=True, trim_spaces=True):
    if uploaded is None:
        return None, "Nenhum arquivo enviado."
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded, dtype=str)
        else:
            df = pd.read_excel(uploaded, dtype=str)
    except Exception as e:
        return None, f"Erro ao ler arquivo: {e}"

    # trim col names
    if trim_spaces:
        df.columns = [c.strip() for c in df.columns]

    # lower mapping
    cols_low = {c: c.lower() for c in df.columns}

    def find_col(poss):
        for p in poss:
            for c, low in cols_low.items():
                if low == p:
                    return c
        # contains
        for p in poss:
            for c, low in cols_low.items():
                if p in low:
                    return c
        return None

    date_col = find_col(["data","date","dia"])
    id_col = find_col(["id","id venda","id_venda","sale id","sale"])
    cliente_col = find_col(["cliente","customer","client"])
    cidade_col = find_col(["cidade"])
    estado_col = find_col(["estado","uf"])
    produto_col = find_col(["produto","product","item","nome"])
    categoria_col = find_col(["categoria","category","cat","tipo"])
    vendedor_col = find_col(["vendedor","seller","salesperson","representante"])
    quantidade_col = find_col(["quantidade","quantity","qtd","qty"])
    preco_col = find_col(["preço unitário","preco unitario","preço","preço_unitario","price","preco"])
    desconto_col = find_col(["desconto","discount"])
    custo_col = find_col(["custo","cost"])
    total_col = find_col(["total","valor_total","valor"])
    lucro_col = find_col(["lucro","margin","profit"])

    # Basic currency cleaning for price/total/cost
    def parse_num(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        s = s.replace("R$", "").replace("$","").replace(" ","")
        if "," in s and "." in s:
            s = s.replace(".","").replace(",",".")
        else:
            if s.count(",") == 1 and s.count(".") == 0:
                s = s.replace(",",".")
        try:
            return float(s)
        except:
            return np.nan

    # parse date
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors='coerce')
        except:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # create normalized df with expected columns
    out = pd.DataFrame()
    out['Data'] = df[date_col] if date_col in df.columns else pd.NaT
    out['ID Venda'] = df[id_col] if id_col in df.columns else df.index.astype(str)
    out['Cliente'] = df[cliente_col] if cliente_col in df.columns else ""
    out['Cidade'] = df[cidade_col] if cidade_col in df.columns else ""
    out['Estado'] = df[estado_col] if estado_col in df.columns else ""
    out['Produto'] = df[produto_col] if produto_col in df.columns else ""
    out['Categoria'] = df[categoria_col] if categoria_col in df.columns else ""
    out['Vendedor'] = df[vendedor_col] if vendedor_col in df.columns else ""
    # numeric columns
    if quantidade_col and quantidade_col in df.columns:
        out['Quantidade'] = pd.to_numeric(df[quantidade_col].apply(lambda x: str(x).replace(",",".")), errors='coerce').fillna(0)
    else:
        out['Quantidade'] = pd.Series([0]*len(out))
    if preco_col and preco_col in df.columns:
        out['Preço Unitário'] = df[preco_col].apply(parse_num)
    else:
        out['Preço Unitário'] = np.nan
    if desconto_col and desconto_col in df.columns:
        out['Desconto'] = df[desconto_col].apply(parse_num)
    else:
        out['Desconto'] = 0.0
    if custo_col and custo_col in df.columns:
        out['Custo'] = df[custo_col].apply(parse_num)
    else:
        out['Custo'] = np.nan
    # total and lucro
    if total_col and total_col in df.columns:
        out['Total'] = df[total_col].apply(parse_num)
    else:
        # compute from qty * price - desconto
        out['Total'] = out['Quantidade'] * out['Preço Unitário']
        out['Total'] = out['Total'].fillna(0) - out['Desconto'].fillna(0)
    if lucro_col and lucro_col in df.columns:
        out['Lucro'] = df[lucro_col].apply(parse_num)
    else:
        # compute if possible
        out['Lucro'] = out['Total'] - out['Custo'].fillna(0)

    # final checks
    if out['Data'].isna().all():
        return None, "Não foi possível detectar/parsear a coluna de datas. Formatos aceitos: YYYY-MM-DD ou DD/MM/YYYY."
    if out['Produto'].isna().all() and out['Categoria'].isna().all():
        return None, "Não foi possível detectar colunas de produto/categoria."

    # Normalize column names to internal friendly ones
    out = out.rename(columns={'Data':'data','ID Venda':'id_venda','Cliente':'cliente','Cidade':'cidade','Estado':'estado',
                              'Produto':'produto','Categoria':'categoria','Vendedor':'vendedor','Quantidade':'quantidade',
                              'Preço Unitário':'preco_unitario','Desconto':'desconto','Custo':'custo','Total':'total','Lucro':'lucro'})
    # Ensure types
    out['quantidade'] = pd.to_numeric(out['quantidade'], errors='coerce').fillna(0)
    out['preco_unitario'] = pd.to_numeric(out['preco_unitario'], errors='coerce')
    out['desconto'] = pd.to_numeric(out['desconto'], errors='coerce').fillna(0)
    out['custo'] = pd.to_numeric(out['custo'], errors='coerce')
    out['total'] = pd.to_numeric(out['total'], errors='coerce').fillna(out['quantidade'] * out['preco_unitario'] - out['desconto'])
    out['lucro'] = pd.to_numeric(out['lucro'], errors='coerce').fillna(out['total'] - out['custo'].fillna(0))

    # add Year/Month cols
    out['year'] = out['data'].dt.year
    out['month'] = out['data'].dt.month
    out['month_name'] = out['data'].dt.strftime("%b")

    return out, None

# -------- Process upload on click ----------
if send_button and uploaded_file is not None:
    df_sales, err = import_sales(uploaded_file, dayfirst=True, trim_spaces=True)
    if err:
        st.sidebar.error(err)
        st.session_state.df = None
    else:
        st.session_state.df = df_sales
        st.session_state.orig_columns = list(df_sales.columns)
        st.sidebar.success("Planilha de vendas carregada com sucesso!")
        
        # 🚨 BLOCO DE DIAGNÓSTICO PARA DADOS VAZIOS 🚨
        if 'produto' in df_sales.columns and df_sales['produto'].astype(str).str.strip().eq('').all():
             st.sidebar.error("DIAGNÓSTICO: A coluna 'produto' parece estar vazia em todos os registros. Verifique o nome da coluna ou os dados no seu arquivo.")
        if 'vendedor' in df_sales.columns and df_sales['vendedor'].astype(str).str.strip().eq('').all():
             st.sidebar.error("DIAGNÓSTICO: A coluna 'vendedor' parece estar vazia em todos os registros. Verifique o nome da coluna ou os dados no seu arquivo.")
        # ------------------------------------------------------------------

# --------- helper: Cohort Analysis ----------
def run_cohort_analysis(df_sales):
    """
    Realiza a Análise de Coorte (Cohort) baseada no mês da primeira compra.
    Retorna um DataFrame que mostra a Receita Média por Cliente por Mês de Retenção.
    """
    if df_sales is None or df_sales.empty:
        return None

    # 1. Encontra o mês da primeira compra para cada cliente (Cohort)
    df_sales['invoice_month'] = df_sales['data'].dt.to_period('M')
    # O 'cliente' é a unidade de coorte
    df_sales['cohort_month'] = df_sales.groupby('cliente')['data'].transform('min').dt.to_period('M')

    # 2. Calcula o número de meses entre o mês da transação e o mês da coorte
    def get_month_difference(series):
        return series.dt.year * 12 + series.dt.month

    invoice_month_num = get_month_difference(df_sales['invoice_month'])
    cohort_month_num = get_month_difference(df_sales['cohort_month'])
    
    # Período de Retenção (Mês 0, Mês 1, Mês 2, etc.)
    df_sales['cohort_index'] = invoice_month_num - cohort_month_num

    # 3. Agrega Receita Total por Cohort e Período de Retenção
    cohort_data = df_sales.groupby(['cohort_month', 'cohort_index'])['total'].sum().reset_index()

    # 4. Conta Clientes Únicos na Coorte (para normalização)
    cohort_sizes = df_sales.groupby('cohort_month')['cliente'].nunique().reset_index(name='cohort_size')

    # 5. Mescla os dados e calcula a Receita Média por Cliente
    cohort_data = pd.merge(cohort_data, cohort_sizes, on='cohort_month')
    cohort_data['avg_revenue'] = cohort_data['total'] / cohort_data['cohort_size']

    # 6. Pivoteia a tabela para o formato de matriz (tabela de Coorte)
    cohort_pivot = cohort_data.pivot_table(index='cohort_month', columns='cohort_index', values='avg_revenue', fill_value=0)
    
    # Formata o índice para YYYY-MM
    cohort_pivot.index = cohort_pivot.index.strftime('%Y-%m')
    
    return cohort_pivot

# --------- Layout main ----------
st.markdown(f"<h1 class='neon-title'>💼 Dashboard Profissional de Vendas</h1>", unsafe_allow_html=True)
st.markdown("Use o menu abaixo para navegar - Detecção de anomalias, relatórios e análise de clientes.")

# Top horizontal menu (ATUALIZADO SEM OPÇÃO CHATGPT)
menu = st.radio("Área", ["Overview", "Vendas", "Produtos", "Vendedores", "Anomalias", "Cohort Analysis", "Exportar / PDF"], horizontal=True)

# -------------- Overview --------------
def kpis_from_sales(df):
    if df is None or df.empty:
        return {}
    total_revenue = df['total'].sum()
    total_qty = df['quantidade'].sum()
    avg_ticket = total_revenue / total_qty if total_qty else np.nan
    avg_price = df['preco_unitario'].mean()
    total_profit = df['lucro'].sum()
    margin = (total_profit / total_revenue * 100) if total_revenue else np.nan
    return {
        "total_revenue": total_revenue,
        "total_qty": total_qty,
        "avg_ticket": avg_ticket,
        "avg_price": avg_price,
        "total_profit": total_profit,
        "margin": margin
    }

if menu == "Overview":
    st.header("Overview - KPIs Rápidos")
    df = st.session_state.df
    if df is None:
        st.info("Envie a planilha de vendas (Modelo B) na barra lateral e clique em Enviar Planilha.")
    else:
        k = kpis_from_sales(df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Receita Total", f"R$ {k['total_revenue']:,.2f}")
        c2.metric("Total Unidades", f"{int(k['total_qty'])}")
        c3.metric("Ticket Médio", f"R$ {k['avg_ticket']:,.2f}" if not np.isnan(k['avg_ticket']) else "N/A")
        c4.metric("Lucro Total", f"R$ {k['total_profit']:,.2f}")
        c5.metric("Margem (%)", f"{k['margin']:.2f}%" if not np.isnan(k['margin']) else "N/A")

        st.markdown("### Receita Mensal")
        monthly = df.groupby(df['data'].dt.to_period("M"))['total'].sum().reset_index()
        monthly['data'] = monthly['data'].astype(str)
        fig = px.line(monthly, x='data', y='total', title="Receita Mensal", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# -------------- Vendas (detailed) --------------
elif menu == "Vendas":
    st.header("Análise de Vendas")
    df = st.session_state.df
    if df is None:
        st.info("Carregue uma planilha de vendas primeiro.")
    else:
        # filters
        col1, col2, col3 = st.columns(3)
        yrs = sorted(df['year'].dropna().unique().astype(int).tolist())
        year_sel = col1.selectbox("Ano", options=yrs, index=len(yrs)-1 if yrs else 0)
        sellers = sorted(df['vendedor'].dropna().unique().tolist())
        seller_sel = col2.multiselect("Vendedor", options=sellers)
        prod_sel = col3.multiselect("Produto", options=sorted(df['produto'].dropna().unique().tolist()))

        dff = df[df['year'] == int(year_sel)]
        if seller_sel:
            dff = dff[dff['vendedor'].isin(seller_sel)]
        if prod_sel:
            dff = dff[dff['produto'].isin(prod_sel)]

        st.subheader("Tabela de Vendas (filtro aplicado)")
        st.dataframe(dff[['data','id_venda','cliente','produto','categoria','vendedor','quantidade','preco_unitario','total','lucro']].sort_values('data', ascending=False).reset_index(drop=True), use_container_width=True)

        st.subheader("Receita por mês (ano selecionado)")
        mon = dff.groupby(dff['data'].dt.month)['total'].sum().reindex(range(1,13), fill_value=0)
        figm = px.bar(x=mon.index, y=mon.values, labels={'x':'Mês','y':'Receita'}, title=f"Receita por mês - {year_sel}")
        st.plotly_chart(figm, use_container_width=True)

        st.subheader("Projeção simples (média móvel 3 meses)")
        rm = monthly = dff.groupby(dff['data'].dt.to_period("M"))['total'].sum().reset_index().sort_values('data')
        if not rm.empty:
            rm['projecao'] = rm['total'].rolling(3, min_periods=1).mean()
            rm['data'] = rm['data'].astype(str)
            figp = px.line(rm, x='data', y=['total','projecao'], title="Receita x Projeção (3m)", markers=True)
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.info("Sem dados suficientes para projeção.")

# -------------- Produtos --------------
elif menu == "Produtos":
    st.header("Top Produtos / Mix")
    df = st.session_state.df
    if df is None:
        st.info("Carregue a planilha.")
    else:
        top_n = st.slider("Top N produtos", 5, 50, 10)
        prod_sum = df.groupby('produto')['total'].sum().sort_values(ascending=False).head(top_n).reset_index()
        fig = px.bar(prod_sum, x='produto', y='total', title=f"Top {top_n} Produtos por Receita")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribuição por Categoria")
        cat_sum = df.groupby('categoria')['total'].sum().reset_index()
        fig2 = px.pie(cat_sum, names='categoria', values='total', title="Receita por Categoria")
        st.plotly_chart(fig2, use_container_width=True)

# -------------- Vendedores --------------
elif menu == "Vendedores":
    st.header("Ranking de Vendedores")
    df = st.session_state.df
    if df is None:
        st.info("Carregue a planilha.")
    else:
        seller_sum = df.groupby('vendedor')['total'].sum().sort_values(ascending=False).reset_index()
        st.dataframe(seller_sum, use_container_width=True)
        fig = px.bar(seller_sum, x='vendedor', y='total', title="Ranking de Vendedores")
        st.plotly_chart(fig, use_container_width=True)

# -------------- Anomalias (Insights Locais) --------------
elif menu == "Anomalias":
    st.header("Detecção de Anomalias (Automático) e Insights Locais")
    df = st.session_state.df
    if df is None:
        st.info("Carregue a planilha para gerar insights.")
    else:
        st.subheader("Detecção de vendas atípicas (z-score sobre total por venda)")
        vals = df[['total']].fillna(0).astype(float)
        scaler = StandardScaler()
        try:
            z = scaler.fit_transform(vals).flatten()
            df['z_score'] = z
            thr = st.slider("Threshold z-score (absoluto) para anomalia", 1.5, 4.0, 2.5)
            anomalies = df[np.abs(df['z_score']) >= thr]
            st.write(f"Encontradas {len(anomalies)} anomalias com |z|>={thr}")
            if not anomalies.empty:
                st.dataframe(anomalies[['data','id_venda','cliente','produto','vendedor','total','z_score']].sort_values('z_score', ascending=False), use_container_width=True)
            else:
                st.info("Nenhuma anomalia detectada com o threshold atual.")
        except Exception as e:
            st.error("Erro na detecção de anomalias: " + str(e))

        st.subheader("Insights automáticos (local)")
        total_rev = df['total'].sum()
        
        # Filtragem e agrupamento robusto para Produtos 
        df_valid_products = df[df['produto'].astype(str).str.strip().astype(bool)]
        top_products = df_valid_products.groupby('produto')['total'].sum().sort_values(ascending=False).head(5)

        # Filtragem e agrupamento robusto para Vendedores 
        df_valid_sellers = df[df['vendedor'].astype(str).str.strip().astype(bool)]
        top_sellers = df_valid_sellers.groupby('vendedor')['total'].sum().sort_values(ascending=False).head(5)

        st.markdown(f"- Receita total: **R$ {total_rev:,.2f}**")
        st.markdown("- Top produtos:")
        for p,v in top_products.items():
            st.markdown(f"  - {p}: R$ {v:,.2f}")
        st.markdown("- Top vendedores:")
        for p,v in top_sellers.items():
            st.markdown(f"  - {p}: R$ {v:,.2f}")

        st.markdown("### Sugestões automáticas (regras simples)")
        suggestions = []
        # Suggest reduce inventory on low-selling items
        low_sales = df_valid_products.groupby('produto')['total'].sum().sort_values().head(3)
        if not low_sales.empty:
            suggestions.append(f"Considere revisar promoção ou substituição para: {', '.join(low_sales.index.tolist())}")
        # High return or negative profit
        bad_profit = df[df['lucro'] < 0]
        if not bad_profit.empty:
            suggestions.append("Há vendas com lucro negativo - revisar preços e custos.")
        # seasonality hint
        monthly = df.groupby(df['data'].dt.to_period("M"))['total'].sum()
        if len(monthly) >= 6:
            recent = monthly.iloc[-3:].mean()
            prev = monthly.iloc[-6:-3].mean() if len(monthly) >= 6 else 0
            if prev > 0 and recent < prev * 0.8:
                suggestions.append("Receita caiu nos últimos 3 meses comparado ao período anterior - revisar campanhas.")
        if suggestions:
            for s in suggestions:
                st.markdown("- " + s)
        else:
            st.markdown("- Nenhuma ação imediata identificada - acompanhe tendências.")

# -------------- Cohort Analysis (NOVA SEÇÃO) --------------
elif menu == "Cohort Analysis":
    st.header("Análise de Cohort (Receita Média por Cliente)")
    df = st.session_state.df
    if df is None:
        st.info("Carregue a planilha para realizar a análise de coorte.")
    else:
        with st.spinner("Calculando Coorte de Receita..."):
            # Passa uma cópia para evitar SettingWithCopyWarning no pandas
            cohort_pivot = run_cohort_analysis(df.copy())
        
        if cohort_pivot is not None:
            st.subheader("Matriz de Receita Média por Cliente (por Cohort)")
            st.markdown("A coorte (linha) representa o **mês da primeira compra**. A coluna representa os **meses desde a primeira compra** (0 = mês da compra). Os valores são a **Receita Média Gerada por Cliente** naquele mês.")
            
            # Exibe a tabela pivotada com formatação
            st.dataframe(cohort_pivot.style.format("R$ {:,.2f}"), use_container_width=True)

            st.subheader("Visualização Heatmap")
            # Cria um heatmap visualmente agradável
            fig_cohort = px.imshow(
                cohort_pivot, 
                text_auto=".2f",
                aspect="auto",
                labels=dict(x="Mês de Retenção (Index)", y="Mês da Cohort (1ª Compra)", color="Receita Média"),
                x=cohort_pivot.columns.tolist(),
                y=cohort_pivot.index.tolist(),
                color_continuous_scale="Viridis"
            )
            fig_cohort.update_layout(title="Heatmap da Receita Média por Cohort e Mês de Retenção")
            st.plotly_chart(fig_cohort, use_container_width=True)

            st.caption("Insights: Procure por linhas (coortes) que mostram uma queda mais lenta na receita média, indicando clientes de maior valor no longo prazo.")
        else:
            st.warning("Não foi possível gerar a análise de coorte. Verifique se as colunas 'data', 'cliente' e 'total' estão presentes e formatadas corretamente.")

# -------------- Export / PDF (BLOCO EXPANDIDO E CORRIGIDO) --------------
elif menu == "Exportar / PDF" or menu == "Exportar / PDF":
    st.header("Export & Relatório")
    df = st.session_state.df
    if df is None:
        st.info("Carregue a planilha antes de exportar.")
    else:
        st.subheader("Exportar dados")
        col1, col2 = st.columns(2)
        col1.download_button("📥 Baixar CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="vendas_export.csv", mime="text/csv")
        with io.BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="vendas")
            col2.download_button("📥 Baixar XLSX", data=buffer.getvalue(), file_name="vendas_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.subheader("Gerar PDF completo")
        logo = st.file_uploader("Logo para PDF (opcional)", type=["png","jpg","jpeg"])
        include_graphs = st.checkbox("Incluir gráficos no PDF (requer 'kaleido')", value=True)
        title_pdf = st.text_input("Título do relatório", value=f"Relatório de Vendas - {datetime.now().year}") 
        if st.button("Gerar PDF"):
            # Garantir que o DataFrame existe
            if df is None:
                st.error("Erro: Nenhum dado carregado para gerar o PDF.")
                st.stop() # SOLUÇÃO CORRETA PARA PARAR O FLUXO NO SCRIPT PRINCIPAL

            # --- 1. PREPARAÇÃO DE DADOS PARA O RELATÓRIO ---
            
            # Recalcula KPIs
            k = kpis_from_sales(df)

            # 1.1. Análise de Anomalias (Recálculo)
            try:
                vals = df[['total']].fillna(0).astype(float)
                scaler = StandardScaler()
                z = scaler.fit_transform(vals).flatten()
                df['z_score'] = z
                # Usamos um threshold padrão (2.5) para o relatório
                anomalies = df[np.abs(df['z_score']) >= 2.5].copy() 
            except:
                anomalies = pd.DataFrame()

            # 1.2. Análise de Cohort (Recálculo)
            cohort_pivot = run_cohort_analysis(df.copy())
            
            # --- 2. GERAÇÃO DE GRÁFICOS ---
            temp_imgs = []
            if include_graphs:
                try:
                    # Gráfico 1: Receita Mensal
                    monthly = df.groupby(df['data'].dt.to_period("M"))['total'].sum().reset_index()
                    monthly['data'] = monthly['data'].astype(str)
                    fig_month = px.line(monthly, x='data', y='total', title='Receita Mensal', markers=True)
                    temp_imgs.append({'title': 'Receita Mensal', 'bytes': fig_month.to_image(format='png', engine='kaleido')})
                    
                    # Gráfico 2: Top Produtos
                    tprod = df.groupby('produto')['total'].sum().reset_index().sort_values('total', ascending=False).head(10)
                    fig_prod = px.bar(tprod, x='produto', y='total', title='Top 10 Produtos por Receita')
                    temp_imgs.append({'title': 'Top 10 Produtos por Receita', 'bytes': fig_prod.to_image(format='png', engine='kaleido')})

                    # Gráfico 3: Heatmap de Cohort (NOVO)
                    if cohort_pivot is not None and not cohort_pivot.empty:
                        fig_cohort = px.imshow(
                            cohort_pivot, 
                            text_auto=".2f",
                            labels=dict(x="Mês de Retenção (Index)", y="Mês da Cohort"),
                            x=cohort_pivot.columns.tolist(),
                            y=cohort_pivot.index.tolist(),
                            color_continuous_scale="Viridis"
                        )
                        temp_imgs.append({'title': 'Heatmap de Receita Média por Cohort', 'bytes': fig_cohort.to_image(format='png', engine='kaleido')})

                except Exception as e:
                    st.warning("Falha ao gerar imagens (kaleido?). Gerando PDF sem gráficos. " + str(e))
                    temp_imgs = []

            # --- 3. INÍCIO DA ESCRITA DO PDF COM LAYOUT CUSTOMIZADO ---
            # Instancia a classe PDF Customizada
            pdf = PDF(primary_color=primary_color) 
            pdf.set_title(title_pdf)
            pdf.alias_nb_pages()
            
            # PÁGINA 1: KPIs e Anomalias
            pdf.add_page()
            
            # Lógica de logo (adicionar a imagem se fornecida)
            if logo:
                try:
                    tmp_logo = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(logo.name)[1])
                    tmp_logo.write(logo.read())
                    tmp_logo.flush()
                    pdf.image(tmp_logo.name, x=10, y=pdf.get_y(), w=30)
                    pdf.ln(20)
                    tmp_logo.close()
                    os.unlink(tmp_logo.name)
                except Exception:
                    pdf.ln(10) 
            else:
                pdf.ln(10)

            # Resumo de KPIs
            pdf.section_title("Resumo de Desempenho")
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 6, f"Receita Total: R$ {k['total_revenue']:,.2f}", ln=True)
            pdf.cell(0, 6, f"Lucro Total: R$ {k['total_profit']:,.2f}", ln=True)
            pdf.cell(0, 6, f"Ticket Médio: R$ {k['avg_ticket']:,.2f}", ln=True)
            pdf.cell(0, 6, f"Margem Bruta: {k['margin']:.2f}%", ln=True)
            pdf.ln(8)

            # Seção Anomalias (com tabela estilizada)
            pdf.section_title(f"Análise de Anomalias (Total de {len(anomalies)} transações atípicas)")
            if not anomalies.empty:
                pdf.set_font("Arial", 'B', 9)
                col_width = [30, 40, 30, 30] 
                
                # Cabeçalho da Tabela
                header_data = ['Data', 'ID Venda', 'Total', 'Z-Score']
                for i, header in enumerate(header_data):
                    pdf.cell(col_width[i], 7, header, 1, 0, 'C')
                pdf.ln()

                pdf.set_font("Arial", '', 9)
                
                # Corpo da Tabela (com cores alternadas)
                for i, row in anomalies[['data', 'id_venda', 'total', 'z_score']].head(10).iterrows():
                    fill = True if i % 2 == 0 else False
                    pdf.set_fill_color(240, 240, 240) if fill else pdf.set_fill_color(255, 255, 255)
                    
                    pdf.cell(col_width[0], 6, row['data'].strftime('%Y-%m-%d'), 1, 0, 'C', fill)
                    pdf.cell(col_width[1], 6, str(row['id_venda']), 1, 0, 'L', fill)
                    pdf.cell(col_width[2], 6, f"R$ {row['total']:,.2f}", 1, 0, 'R', fill)
                    pdf.cell(col_width[3], 6, f"{row['z_score']:.2f}", 1, 0, 'C', fill)
                    pdf.ln()

            else:
                pdf.cell(0, 6, "Nenhuma anomalia significativa detectada (Z-score > 2.5).", 0, 1)
            
            
            # --- 5. ADIÇÃO DE GRÁFICOS E COHORTS ---
            
            # Insere gráficos (cada um em uma nova página)
            for img_data in temp_imgs:
                t = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                t.write(img_data['bytes'])
                t.flush()
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 8, img_data['title'], ln=True, align='C')
                pdf.ln(4)
                try:
                    pdf.image(t.name, x=15, y=pdf.get_y(), w=180) 
                except Exception:
                    pass
                t.close()
                os.unlink(t.name)


            # --- 6. EXPORTAÇÃO FINAL (RESOLVENDO O TYPE ERROR) ---
            out = "relatorio_vendas.pdf"
            
            # 1. Obter a saída do PDF 
            pdf_output_data = pdf.output(dest='S')
            
            # 2. 🌟 SOLUÇÃO DEFINITIVA: Garante que é um objeto de bytes 🌟
            if isinstance(pdf_output_data, str):
                # Se for string (versão antiga), codifica para bytes com latin-1
                pdf_output_bytes = pdf_output_data.encode('latin-1')
            else:
                # Se já for bytes (fpdf2 ou ambiente correto), usa o objeto diretamente
                pdf_output_bytes = pdf_output_data
            
            # 3. Escreve a saída de bytes no arquivo
            with open(out, "wb") as f:
                f.write(pdf_output_bytes)
            
            with open(out, "rb") as f:
                st.download_button("📥 Baixar PDF", data=f, file_name=out, mime="application/pdf")
            st.success("PDF gerado!")

# ----------------- End -----------------
st.caption("Feito com ❤️ - peça ajustes, templates PDF personalizados ou integração com Google Sheets/DB.")