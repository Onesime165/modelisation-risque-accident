import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_curve
import statsmodels.api as sm
import math
import warnings
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="Analyse Accidents Routiers",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé - Design sombre technologique avec touches lumineuses
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3e 25%, #0f1423 50%, #1e2645 75%, #0d1128 100%);
        background-attachment: fixed;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(ellipse at 20% 30%, rgba(0, 212, 255, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 70%, rgba(138, 43, 226, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(0, 255, 157, 0.08) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }
    
    .main-title {
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        color: #00d4ff;
        font-size: 3rem;
        font-weight: 900;
        text-shadow: 
            0 0 10px rgba(0, 212, 255, 0.8),
            0 0 20px rgba(0, 212, 255, 0.6),
            0 0 30px rgba(0, 212, 255, 0.4),
            0 0 40px rgba(0, 212, 255, 0.2);
        padding: 30px;
        margin-bottom: 40px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        border-top: 1px solid rgba(0, 212, 255, 0.3);
        border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 10px rgba(0, 212, 255, 0.8), 0 0 20px rgba(0, 212, 255, 0.6); }
        to { text-shadow: 0 0 20px rgba(0, 212, 255, 1), 0 0 30px rgba(0, 212, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.6); }
    }
    
    .section-title {
        font-family: 'Orbitron', sans-serif;
        color: #00ff9d;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 40px;
        margin-bottom: 25px;
        padding: 15px 20px;
        background: linear-gradient(135deg, rgba(0, 255, 157, 0.15) 0%, rgba(0, 212, 255, 0.15) 100%);
        border-left: 5px solid #00ff9d;
        border-radius: 0 10px 10px 0;
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.2);
        position: relative;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        right: 0;
        top: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, transparent, #00ff9d, transparent);
    }
    
    .metric-container {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(138, 43, 226, 0.1) 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(0, 212, 255, 0.4);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
        backdrop-filter: blur(10px);
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.25);
        border-color: rgba(0, 255, 157, 0.6);
    }
    
    .conclusion-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.12) 0%, rgba(138, 43, 226, 0.12) 50%, rgba(0, 255, 157, 0.12) 100%);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(0, 212, 255, 0.3);
        margin: 25px 0;
        color: #e0e0e0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 0 20px rgba(0, 212, 255, 0.05);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .conclusion-box::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00d4ff, #8a2be2, #00ff9d, #00d4ff);
        border-radius: 15px;
        opacity: 0;
        z-index: -1;
        transition: opacity 0.3s ease;
        animation: borderRotate 3s linear infinite;
    }
    
    .conclusion-box:hover::before {
        opacity: 0.3;
    }
    
    @keyframes borderRotate {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .conclusion-box h3, .conclusion-box h4 {
        color: #00d4ff;
        font-family: 'Rajdhani', sans-serif;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .stButton>button {
        font-family: 'Rajdhani', sans-serif;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 50%, #00ff9d 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        border-radius: 10px;
        padding: 12px 35px;
        transition: all 0.4s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: scale(1.08);
        box-shadow: 0 8px 30px rgba(0, 255, 157, 0.6);
    }
    
    .stDataFrame {
        background: rgba(10, 14, 39, 0.6);
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(10, 14, 39, 0.5);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        color: #00d4ff;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.2);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.3), rgba(0, 255, 157, 0.3));
        border-color: #00ff9d;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(138, 43, 226, 0.15));
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: #00ff9d;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.95) 0%, rgba(26, 29, 62, 0.95) 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(10, 14, 39, 0.98) 0%, 
            rgba(18, 22, 50, 0.98) 25%,
            rgba(26, 29, 62, 0.98) 50%,
            rgba(18, 22, 50, 0.98) 75%,
            rgba(10, 14, 39, 0.98) 100%);
        border-right: 3px solid transparent;
        border-image: linear-gradient(180deg, 
            transparent, 
            rgba(0, 212, 255, 0.8), 
            rgba(138, 43, 226, 0.8),
            rgba(0, 255, 157, 0.8),
            transparent) 1;
        box-shadow: 
            inset 0 0 30px rgba(0, 212, 255, 0.1),
            5px 0 20px rgba(0, 0, 0, 0.5);
        position: relative;
    }
    
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(ellipse at 50% 20%, rgba(0, 212, 255, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(138, 43, 226, 0.15) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
        position: relative;
        z-index: 1;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        font-family: 'Rajdhani', sans-serif;
        color: #00d4ff !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        margin-bottom: 20px !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        gap: 8px;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(138, 43, 226, 0.1));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 12px 15px !important;
        margin: 5px 0 !important;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        color: #c0c0c0 !important;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
        transition: left 0.5s ease;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(138, 43, 226, 0.2));
        border-color: rgba(0, 255, 157, 0.6);
        color: #00ff9d !important;
        transform: translateX(5px);
        box-shadow: 
            0 0 15px rgba(0, 212, 255, 0.4),
            inset 0 0 10px rgba(0, 212, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover::before {
        left: 100%;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        background-color: rgba(0, 212, 255, 0.3) !important;
        border-color: #00d4ff !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label > div:first-child > div {
        background-color: #00ff9d !important;
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.8);
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.3), rgba(0, 255, 157, 0.3));
        border: 2px solid #00ff9d;
        color: #00ff9d !important;
        font-weight: 700 !important;
        box-shadow: 
            0 0 20px rgba(0, 255, 157, 0.5),
            inset 0 0 20px rgba(0, 212, 255, 0.2);
        transform: translateX(8px);
    }
    
    [data-testid="stSidebar"] h2 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00d4ff !important;
        text-align: center !important;
        font-size: 1.5rem !important;
        padding: 15px 10px !important;
        margin-bottom: 25px !important;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent);
        border-top: 1px solid rgba(0, 212, 255, 0.5);
        border-bottom: 1px solid rgba(0, 212, 255, 0.5);
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.8);
    }
    
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        color: #e0e0e0;
    }
    
    p, li {
        color: #c0c0c0;
        line-height: 1.8;
    }
    
    .stAlert {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-title">🚗 Analyse et modélisation du risque d\'accident par régression logistique</h1>', unsafe_allow_html=True)

# Chargement et préparation des données
@st.cache_data
def load_and_prepare_data():
    df = pd.read_excel("dataset_accident.xlsx")
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(include='object'):
        mode = df[col].mode()[0]
        df[col].fillna(mode, inplace=True)
    for col in df.select_dtypes(include='number'):
        df[col] = winsorize(df[col], limits=[0.05, 0.05])
    return df

@st.cache_resource
def build_model(df):
    X = pd.get_dummies(df.drop(columns=["Accident"]), drop_first=True)
    X = X.astype(float)
    X = sm.add_constant(X)
    y = df["Accident"].map({"Non": 0, "Oui": 1}).astype(float)
    model = sm.Logit(y, X).fit(disp=0)
    return model, X, y

try:
    df = load_and_prepare_data()
    model, X, y = build_model(df)
    
    # Sidebar pour la navigation
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio(
        "Choisir une section",
        ["🏠 Introduction", "📈 Exploration des données", "📊 Statistiques descriptives", 
         "🔬 Tests statistiques", "🤖 Régression logistique", "🎯 Prédiction", "📝 Conclusion"]
    )
    
    # SECTION 1: INTRODUCTION
    if section == "🏠 Introduction":
        st.markdown('<h2 class="section-title">1. Introduction</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="conclusion-box">
        <h3>📌 Contexte de l'étude</h3>
        <p>Chaque année, le nombre d'accidents de la route continue de constituer une préoccupation majeure 
        en matière de santé publique et de sécurité routière. L'impact de ces événements dépasse largement 
        les seuls dommages humains, engendrant aussi des coûts économiques et sociaux importants.</p>
        
        <h4>🎯 Problématique</h4>
        <p><strong>Quels sont les facteurs significativement associés à la survenue d'un accident et comment 
        peut-on modéliser le risque d'accident en fonction de ces facteurs pour améliorer les stratégies de prévention ?</strong></p>
        
        <h4>📋 Données analysées</h4>
        <ul>
            <li><strong>840 observations</strong> détaillées sur les accidents routiers</li>
            <li><strong>14 variables</strong> : 4 quantitatives et 10 qualitatives</li>
            <li>Facteurs météorologiques, caractéristiques des routes, comportement des conducteurs, densité du trafic, type de véhicule</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Observations totales", "840")
        with col2:
            st.metric("📈 Variables étudiées", "14")
        with col3:
            st.metric("🎯 Taux d'accidents", f"{(df['Accident']=='Oui').mean()*100:.1f}%")
    
    # SECTION 2: EXPLORATION DES DONNÉES
    elif section == "📈 Exploration des données":
        st.markdown('<h2 class="section-title">2. Exploration des données</h2>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📋 Aperçu des données", "🧹 Traitement"])
        
        with tab1:
            st.markdown("### 📊 Premières lignes du dataset")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("### 📈 Informations générales")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Nombre de lignes :** {df.shape[0]}")
                st.info(f"**Variables quantitatives :** {df.select_dtypes(include='number').shape[1]}")
            with col2:
                st.info(f"**Nombre de colonnes :** {df.shape[1]}")
                st.info(f"**Variables qualitatives :** {df.select_dtypes(include='object').shape[1]}")
        
        with tab2:
            st.markdown("### 🧹 Traitement effectué")
            st.markdown("""
            <div class="conclusion-box">
            <h4>✅ Étapes de nettoyage réalisées :</h4>
            <ul>
                <li><strong>Valeurs manquantes :</strong> 5% global (42 valeurs par variable)
                    <ul>
                        <li>Imputation par <strong>médiane</strong> pour les variables quantitatives</li>
                        <li>Imputation par <strong>mode</strong> pour les variables qualitatives</li>
                    </ul>
                </li>
                <li><strong>Valeurs aberrantes :</strong> Traitement par <strong>winsorisation</strong> (5% de chaque côté)</li>
                <li><strong>Résultat :</strong> Base de données propre et prête pour l'analyse statistique (798 observations complètes)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # SECTION 3: STATISTIQUES DESCRIPTIVES
    elif section == "📊 Statistiques descriptives":
        st.markdown('<h2 class="section-title">3. Statistiques descriptives</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Variables quantitatives", "📈 Variables qualitatives", "🔄 Analyse croisée"])
        
        with tab1:
            st.markdown("### 📊 Résumé statistique des variables quantitatives")
            st.dataframe(df.describe().transpose().round(2), use_container_width=True)
            
            st.markdown("### 📈 Distributions des variables quantitatives")
            quant_vars = ["Limitation_de_vitesse", "Nombre_de_véhicules", 
                         "Age_du_conducteur", "Experience_du_conducteur"]
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.patch.set_facecolor('#0a0e27')
            axes = axes.flatten()
            colors = ["#00d4ff", "#ff6b35", "#00ff9d", "#8a2be2"]
            
            for i, (var, color) in enumerate(zip(quant_vars, colors)):
                axes[i].set_facecolor('#1a1d3e')
                sns.histplot(df[var], kde=True, bins=20, ax=axes[i], color=color, edgecolor='white', alpha=0.7)
                axes[i].set_title(f'Distribution de {var}', color='#00ff9d', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(var, color='white')
                axes[i].set_ylabel('Fréquence', color='white')
                axes[i].tick_params(colors='white')
                for spine in axes[i].spines.values():
                    spine.set_color('#00d4ff')
                    spine.set_linewidth(1.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>🔍 Observations clés :</h4>
            <ul>
                <li><strong>Limitation de vitesse moyenne :</strong> 68 km/h</li>
                <li><strong>Nombre moyen de véhicules impliqués :</strong> 3 véhicules</li>
                <li><strong>Âge moyen du conducteur :</strong> 43 ans</li>
                <li><strong>Expérience moyenne de conduite :</strong> 39 ans</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 📊 Résumé statistique des variables qualitatives")
            st.dataframe(df.select_dtypes(include='object').describe().transpose(), use_container_width=True)
            
            st.markdown("### 📈 Répartition de toutes les variables qualitatives")
            qual_vars = ["Meteo", "Type_de_route", "Heure_du_jour", "Densité_du_trafic",
                        "Conducteur_Alcool", "Gravité_de_l'accident", "Etat_de_la_route",
                        "Type_de_vehicule", "condition_eclairage_route", "Accident"]
            
            n_cols = 3
            n_rows = math.ceil(len(qual_vars) / n_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
            fig.patch.set_facecolor('#0a0e27')
            axes = axes.flatten()
            
            colors_palette = ['#00d4ff', '#8a2be2', '#00ff9d', '#ff6b35', '#ffd700']
            
            for i, var in enumerate(qual_vars):
                axes[i].set_facecolor('#1a1d3e')
                value_counts = df[var].value_counts()
                bars = axes[i].bar(range(len(value_counts)), value_counts.values, 
                                  color=colors_palette[i % len(colors_palette)], 
                                  edgecolor='white', linewidth=1.5, alpha=0.8)
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right', color='white')
                axes[i].set_title(f'Répartition de {var}', color='#00ff9d', fontsize=11, fontweight='bold')
                axes[i].set_ylabel("Nombre d'observations", color='white')
                axes[i].tick_params(colors='white')
                
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', color='white', fontsize=9)
                
                for spine in axes[i].spines.values():
                    spine.set_color('#00d4ff')
                    spine.set_linewidth(1.5)
            
            for j in range(len(qual_vars), len(axes)):
                fig.delaxes(axes[j])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>🔍 Observations principales :</h4>
            <ul>
                <li><strong>Météo :</strong> 45% des cas sous conditions claires</li>
                <li><strong>Type de route :</strong> 52% sur autoroute</li>
                <li><strong>Densité du trafic :</strong> 41% avec densité modérée</li>
                <li><strong>Consommation d'alcool :</strong> 85% sans alcool</li>
                <li><strong>État de la route :</strong> 52% sur route sèche</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 🔄 Analyse croisée : Accident vs Variables quantitatives")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.patch.set_facecolor('#0a0e27')
            axes = axes.flatten()
            
            for i, var in enumerate(quant_vars):
                axes[i].set_facecolor('#1a1d3e')
                sns.boxplot(data=df, x="Accident", y=var, ax=axes[i], 
                           palette={'Non': '#00d4ff', 'Oui': '#ff6b35'})
                axes[i].set_title(f'{var} selon la survenue d\'un accident', 
                                color='#00ff9d', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Accident', color='white')
                axes[i].set_ylabel(var, color='white')
                axes[i].tick_params(colors='white')
                for spine in axes[i].spines.values():
                    spine.set_color('#00d4ff')
                    spine.set_linewidth(1.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("### 📊 Tableau des moyennes par groupe")
            moyennes_df = df.groupby("Accident")[quant_vars].mean().round(2)
            st.dataframe(moyennes_df, use_container_width=True)
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>🔍 Parmi les personnes accidentées :</h4>
            <ul>
                <li><strong>Limitation de vitesse moyenne :</strong> 67 km/h</li>
                <li><strong>Nombre moyen de véhicules impliqués :</strong> 3</li>
                <li><strong>Âge moyen des conducteurs :</strong> 44 ans</li>
                <li><strong>Expérience moyenne du conducteur :</strong> 40 ans</li>
            </ul>
            <p><em>À travers ces boxplots, on peut supposer qu'il n'existe pas de liaison forte entre les variables quantitatives et la variable Accident.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔄 Analyse croisée : Accident vs Variables qualitatives")
            
            qual_vars_analyse = ["Meteo", "Type_de_route", "Heure_du_jour", "Densité_du_trafic",
                                "Conducteur_Alcool", "Gravité_de_l'accident", "Etat_de_la_route",
                                "Type_de_vehicule", "condition_eclairage_route"]
            
            n_cols = 3
            n_rows = math.ceil(len(qual_vars_analyse) / n_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
            fig.patch.set_facecolor('#0a0e27')
            axes = axes.flatten()
            
            for i, var in enumerate(qual_vars_analyse):
                axes[i].set_facecolor('#1a1d3e')
                ct = pd.crosstab(df[var], df["Accident"])
                ct.plot(kind='bar', ax=axes[i], color=['#00d4ff', '#ff6b35'], 
                       edgecolor='white', linewidth=1.5, alpha=0.8)
                axes[i].set_title(f'{var} vs Accident', color='#00ff9d', 
                                fontsize=11, fontweight='bold')
                axes[i].set_xlabel('', color='white')
                axes[i].set_ylabel("Nombre d'observations", color='white')
                axes[i].tick_params(colors='white', rotation=45)
                axes[i].legend(['Non', 'Oui'], title='Accident', 
                             facecolor='#1a1d3e', edgecolor='#00d4ff')
                
                for container in axes[i].containers:
                    axes[i].bar_label(container, fmt='%d', label_type='edge', 
                                    fontsize=8, color='white')
                
                for spine in axes[i].spines.values():
                    spine.set_color('#00d4ff')
                    spine.set_linewidth(1.5)
            
            for j in range(len(qual_vars_analyse), len(axes)):
                fig.delaxes(axes[j])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("### 📊 Tableaux croisés (proportions)")
            
            col1, col2 = st.columns(2)
            
            for idx, var in enumerate(qual_vars_analyse):
                with col1 if idx % 2 == 0 else col2:
                    st.markdown(f"#### {var}")
                    ct = pd.crosstab(df[var], df["Accident"], normalize='index').round(2)
                    st.dataframe(ct, use_container_width=True)
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>🔍 Parmi les personnes ayant eu un accident :</h4>
            <ul>
                <li><strong>38%</strong> étaient exposées à des conditions météorologiques orageuses</li>
                <li><strong>34%</strong> circulaient sur une route rurale</li>
                <li><strong>31%</strong> ont eu leur accident en soirée ou durant la nuit</li>
                <li><strong>29%</strong> évoluaient dans un trafic de densité modérée</li>
                <li><strong>30%</strong> avaient consommé de l'alcool</li>
                <li><strong>31%</strong> ont subi un accident de gravité modérée</li>
                <li><strong>32%</strong> ont eu un accident sur une chaussée sèche</li>
                <li><strong>35%</strong> étaient impliquées dans un accident avec un camion</li>
                <li><strong>29%</strong> ont eu un accident dans des conditions d'éclairage comprenant soit la lumière du jour, soit l'absence totale de lumière</li>
            </ul>
            <p><strong>Conclusion :</strong> L'analyse descriptive confirme des caractéristiques cohérentes avec un âge moyen de 43 ans, 
            une expérience moyenne de 39 ans, et 3 véhicules impliqués en moyenne. Ces observations posent les bases pour 
            la recherche d'associations statistiques plus spécifiques.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # SECTION 4: TESTS STATISTIQUES
    elif section == "🔬 Tests statistiques":
        st.markdown('<h2 class="section-title">4. Tests statistiques</h2>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔬 Tests Chi² (Variables qualitatives)", "📊 Tests Kruskal-Wallis (Variables quantitatives)"])
        
        with tab1:
            st.markdown("### 🔬 Tests d'association Chi² avec la variable Accident")
            
            qual_vars = ["Meteo", "Type_de_route", "Heure_du_jour", "Densité_du_trafic",
                        "Conducteur_Alcool", "Gravité_de_l'accident", "Etat_de_la_route",
                        "Type_de_vehicule", "condition_eclairage_route"]
            
            results = []
            for var in qual_vars:
                table = pd.crosstab(df["Accident"], df[var])
                chi2_stat, p_value, dof, expected = chi2_contingency(table)
                results.append({
                    "Variable": var,
                    "Chi²": f"{chi2_stat:.4f}",
                    "ddl": dof,
                    "p-value": f"{p_value:.4f}",
                    "Significatif (α=0.05)": "✅ Oui" if p_value < 0.05 else "❌ Non"
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>🔍 Résultats des tests :</h4>
            <ul>
                <li><strong>✅ Liaison significative détectée :</strong> État de la route (p = 0.0239)</li>
                <li><strong>❌ Aucune liaison significative :</strong> Météo (p = 0.0573), Type de route (p = 0.2715), 
                Heure du jour (p = 0.6102), Densité du trafic (p = 0.9516), Conducteur alcool (p = 0.8181), 
                Gravité (p = 0.4994), Type de véhicule (p = 0.3647), Éclairage (p = 0.9789)</li>
                <li><strong>📊 Interprétation :</strong> Seul l'<strong>état de la route</strong> montre une association 
                statistiquement significative avec la survenue d'accidents au seuil de 5%</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 📊 Tests de Kruskal-Wallis pour variables quantitatives")
            
            from scipy.stats import kruskal
            quant_vars = ["Limitation_de_vitesse", "Nombre_de_véhicules",
                         "Age_du_conducteur", "Experience_du_conducteur"]
            
            results_quant = []
            for var in quant_vars:
                groups = [df[df["Accident"] == cat][var].dropna() for cat in df["Accident"].unique()]
                stat, p = kruskal(*groups)
                results_quant.append({
                    "Variable": var,
                    "Statistique H": f"{stat:.4f}",
                    "p-value": f"{p:.4f}",
                    "Significatif (α=0.05)": "✅ Oui" if p < 0.05 else "❌ Non"
                })
            
            results_quant_df = pd.DataFrame(results_quant)
            st.dataframe(results_quant_df, use_container_width=True)
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>📌 Conclusion partielle des tests statistiques :</h4>
            <p>Les tests d'association ont révélé une seule liaison statistiquement significative entre la survenue 
            d'accident et l'<strong>état de la route</strong> (p=0.024), particulièrement les routes sèches qui 
            augmentent les risques comparé aux routes en construction.</p>
            <p>Aucune association significative n'a été détectée avec les autres variables qualitatives (météo, 
            type de route, alcool, gravité, éclairage) ni avec les variables quantitatives (âge, expérience, 
            vitesse limite, nombre de véhicules).</p>
            <p><strong>Ces résultats suggèrent que l'état de la route est un facteur clé influençant les accidents, 
            nécessitant une modélisation plus fine.</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # SECTION 5: RÉGRESSION LOGISTIQUE
    elif section == "🤖 Régression logistique":
        st.markdown('<h2 class="section-title">5. Régression logistique</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Modèle", "📈 Odds Ratios", "🎯 Performance", "🔍 Diagnostic"])
        
        with tab1:
            st.markdown("### 🤖 Résultats du modèle de régression logistique")
            
            coefs = pd.DataFrame({
                'Coefficient': model.params,
                'Erreur std': model.bse,
                'z': model.tvalues,
                'p-value': model.pvalues,
                'Significatif': model.pvalues < 0.05
            }).sort_values('p-value')
            
            st.dataframe(coefs.head(15), use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pseudo R²", f"{model.prsquared:.4f}")
            with col2:
                st.metric("AIC", f"{model.aic:.2f}")
            with col3:
                st.metric("Log-Likelihood", f"{model.llf:.2f}")
            with col4:
                st.metric("Observations", len(df))
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>🔍 Analyse du modèle :</h4>
            <p>La constante et la modalité <strong>Sec</strong> de la variable <strong>Etat_de_la_route</strong> 
            sont les seules significatives au seuil de 5%.</p>
            <p>Le modèle présente un Pseudo R² de 0.033, indiquant une capacité prédictive limitée.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 📈 Odds Ratios (Rapports de cotes)")
            
            odds_ratios = np.exp(model.params).sort_values(ascending=False)
            or_df = pd.DataFrame({
                'Variable': odds_ratios.index,
                'Odds Ratio': odds_ratios.values,
                'Interprétation': ['Augmentation du risque' if x > 1 else 'Diminution du risque' for x in odds_ratios.values]
            }).head(15)
            
            st.dataframe(or_df, use_container_width=True)
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>🔍 Interprétations clés des Odds Ratios :</h4>
            <ul>
                <li><strong>Etat_de_la_route_Sec : OR = 1.78</strong><br>
                Par rapport à une route en construction, une route sèche multiplie par 1.78 les chances d'avoir un accident.
                Cela représente une augmentation de 78% du risque.</li>
                <li><strong>Constante : OR = 0.18</strong><br>
                Toutes choses étant égales par ailleurs, la probabilité de base qu'un accident survienne 
                dans les conditions de référence.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Effets marginaux")
            mfx = model.get_margeff()
            mfx_df = pd.DataFrame({
                'Variable': mfx.summary_frame().index,
                'dy/dx': mfx.summary_frame()['dy/dx'],
                'p-value': mfx.summary_frame()['Pr(>|z|)']
            }).sort_values('p-value').head(10)
            
            st.dataframe(mfx_df, use_container_width=True)
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>🔍 Interprétation des effets marginaux :</h4>
            <p>Par rapport à un état de la route en construction, la probabilité de risque d'accident 
            augmente d'environ <strong>0.113</strong> (soit 11.3 points de pourcentage) pour une route sèche.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 🎯 Performance du modèle")
            
            df_temp = df.copy()
            df_temp["PROBABILITE_PREDITE"] = model.predict()
            
            fpr, tpr, thresholds = roc_curve(
                df_temp["Accident"].map({"Non": 0, "Oui": 1}),
                df_temp["PROBABILITE_PREDITE"]
            )
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[optimal_idx]
            
            df_temp["MODALITE_PREDITE"] = np.where(
                df_temp["PROBABILITE_PREDITE"] < optimal_threshold, "Non", "Oui"
            )
            
            matrice = pd.crosstab(df_temp["Accident"], df_temp["MODALITE_PREDITE"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Matrice de confusion")
                st.dataframe(matrice, use_container_width=True)
            
            with col2:
                VP = matrice.loc["Oui", "Oui"] if "Oui" in matrice.columns else 0
                VN = matrice.loc["Non", "Non"] if "Non" in matrice.columns else 0
                FP = matrice.loc["Non", "Oui"] if "Oui" in matrice.columns else 0
                FN = matrice.loc["Oui", "Non"] if "Non" in matrice.columns else 0
                
                tmc = (FP + FN) / (VP + VN + FP + FN)
                accuracy = (VP + VN) / (VP + VN + FP + FN)
                
                st.metric("🎯 Seuil optimal", f"{optimal_threshold:.2f}")
                st.metric("✅ Précision globale", f"{accuracy*100:.2f}%")
                st.metric("❌ Taux de mauvais classement", f"{tmc*100:.2f}%")
                
                if VP + FN > 0:
                    sensibilite = VP / (VP + FN)
                    st.metric("📈 Sensibilité", f"{sensibilite*100:.2f}%")
                
                if VN + FP > 0:
                    specificite = VN / (VN + FP)
                    st.metric("📉 Spécificité", f"{specificite*100:.2f}%")
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>📊 Analyse de performance :</h4>
            <ul>
                <li><strong>Taux de mauvais classement :</strong> ~35% (impliquant une marge importante d'erreur)</li>
                <li><strong>Seuil optimal :</strong> 0.32 (déterminé par l'indice de Youden)</li>
                <li><strong>Précision :</strong> ~65% (Le modèle classe correctement environ 2/3 des observations)</li>
                <li><strong>Conclusion :</strong> Le modèle présente une performance modérée. Une optimisation 
                avec des variables supplémentaires ou des méthodes avancées pourrait améliorer les résultats.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 🔍 Diagnostic des résidus")
            
            influence = model.get_influence()
            res_studentized = influence.resid_studentized
            
            fig, ax = plt.subplots(figsize=(14, 5))
            fig.patch.set_facecolor('#0a0e27')
            ax.set_facecolor('#1a1d3e')
            
            ax.plot(res_studentized, marker='o', linestyle='none', markersize=4, 
                   color='#00d4ff', alpha=0.6)
            ax.axhline(y=0, linestyle='-', color='#00ff9d', linewidth=2)
            ax.axhline(y=2, linestyle='--', color='#ff6b35', linewidth=2, label='Seuils ±2')
            ax.axhline(y=-2, linestyle='--', color='#ff6b35', linewidth=2)
            ax.axhline(y=3, linestyle=':', color='red', linewidth=1.5, alpha=0.7, label='Seuils ±3')
            ax.axhline(y=-3, linestyle=':', color='red', linewidth=1.5, alpha=0.7)
            ax.set_ylim(-4, 4)
            ax.set_ylabel("Résidus studentisés", color='white', fontsize=12)
            ax.set_xlabel("Index des observations", color='white', fontsize=12)
            ax.set_title("Diagnostic des résidus studentisés", color='#00ff9d', 
                        fontsize=14, fontweight='bold')
            ax.tick_params(colors='white')
            ax.legend(facecolor='#1a1d3e', edgecolor='#00d4ff', labelcolor='white')
            
            for spine in ax.spines.values():
                spine.set_color('#00d4ff')
                spine.set_linewidth(1.5)
            
            ax.grid(True, alpha=0.2, color='white', linestyle=':')
            
            st.pyplot(fig)
            
            pourcentage_2 = (np.abs(res_studentized) <= 2).mean() * 100
            pourcentage_3 = (np.abs(res_studentized) <= 3).mean() * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ {pourcentage_2:.2f}% des résidus dans l'intervalle [-2, 2]")
            with col2:
                st.success(f"✅ {pourcentage_3:.2f}% des résidus dans l'intervalle [-3, 3]")
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>📊 Analyse des résidus :</h4>
            <p>En théorie, 95% des résidus studentisés devraient se trouver dans l'intervalle [-2, 2].</p>
            <p>L'analyse montre que <strong>100%</strong> des observations sont dans les limites acceptables [-3, 3], 
            ce qui indique une bonne qualité d'ajustement au niveau des résidus. Toutefois, le faible Pseudo R² 
            suggère qu'une optimisation du modèle reste possible avec l'ajout de variables explicatives pertinentes.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="conclusion-box">
            <h4>📌 Conclusion partielle - Régression Logistique :</h4>
            <p>Le modèle de prédiction présente un <strong>taux de mauvais classement de ~35%</strong>, 
            impliquant une marge importante d'erreur. L'analyse des résidus met en lumière une bonne modélisation 
            pour 100% des observations au sens des résidus studentisés, mais suggère qu'une optimisation du modèle 
            est possible.</p>
            <p>Le <strong>seuil optimal</strong> pour prédire la survenue d'accident a été déterminé selon l'indice 
            de Youden à <strong>0.32</strong>, facilitant ainsi la classification binaire des cas.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # SECTION 6: PRÉDICTION
    elif section == "🎯 Prédiction":
        st.markdown('<h2 class="section-title">6. Prédiction du risque d\'accident</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="conclusion-box">
        <h4>🎯 Outil de prédiction interactif</h4>
        <p>Renseignez les caractéristiques d'une situation pour estimer le risque d'accident selon le modèle développé.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🌤️ Conditions environnementales")
            meteo = st.selectbox("Météo", sorted(df["Meteo"].unique()))
            type_route = st.selectbox("Type de route", sorted(df["Type_de_route"].unique()))
            heure = st.selectbox("Heure du jour", sorted(df["Heure_du_jour"].unique()))
            densite = st.selectbox("Densité du trafic", sorted(df["Densité_du_trafic"].unique()))
            etat_route = st.selectbox("État de la route", sorted(df["Etat_de_la_route"].unique()))
        
        with col2:
            st.markdown("#### 🚗 Caractéristiques véhicule/route")
            vitesse = st.slider("Limitation de vitesse (km/h)", 30, 130, 60, 10)
            nb_vehicules = st.slider("Nombre de véhicules", 1, 5, 3)
            type_vehicule = st.selectbox("Type de véhicule", sorted(df["Type_de_vehicule"].unique()))
            eclairage = st.selectbox("Éclairage route", sorted(df["condition_eclairage_route"].unique()))
            gravite = st.selectbox("Gravité potentielle", sorted(df["Gravité_de_l'accident"].unique()))
        
        with col3:
            st.markdown("#### 👤 Caractéristiques du conducteur")
            age = st.slider("Âge du conducteur", 18, 70, 40)
            experience = st.slider("Expérience (années)", 0, 50, 20)
            alcool = st.selectbox("Consommation d'alcool", ["Non", "Oui"])
        
        if st.button("🎯 PRÉDIRE LE RISQUE D'ACCIDENT", use_container_width=True):
            nouvel_individu = pd.DataFrame([{
                "Meteo": meteo,
                "Type_de_route": type_route,
                "Heure_du_jour": heure,
                "Densité_du_trafic": densite,
                "Limitation_de_vitesse": float(vitesse),
                "Nombre_de_véhicules": float(nb_vehicules),
                "Conducteur_Alcool": alcool,
                "Gravité_de_l'accident": gravite,
                "Etat_de_la_route": etat_route,
                "Type_de_vehicule": type_vehicule,
                "Age_du_conducteur": float(age),
                "Experience_du_conducteur": float(experience),
                "condition_eclairage_route": eclairage
            }])
            
            ind_encoded = pd.get_dummies(nouvel_individu)
            model_vars = model.model.exog_names[1:]
            
            for col in model_vars:
                if col not in ind_encoded.columns:
                    ind_encoded[col] = 0
            
            ind_encoded = ind_encoded[model_vars]
            ind_encoded = sm.add_constant(ind_encoded, has_constant='add')
            ind_encoded = ind_encoded.astype(float)
            
            proba = model.predict(ind_encoded)[0]
            modalite = "Oui" if proba >= 0.32 else "Non"
            
            st.markdown("---")
            st.markdown("### 📊 Résultats de la prédiction")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎲 Probabilité d'accident", f"{proba*100:.2f}%")
            with col2:
                risk_icon = "🔴" if modalite == "Oui" else "🟢"
                st.metric(f"{risk_icon} Risque d'accident", modalite)
            with col3:
                risk_level = "ÉLEVÉ" if proba >= 0.5 else "MODÉRÉ" if proba >= 0.32 else "FAIBLE"
                st.metric("📈 Niveau de risque", risk_level)
            
            fig, ax = plt.subplots(figsize=(12, 2))
            fig.patch.set_facecolor('#0a0e27')
            ax.set_facecolor('#1a1d3e')
            
            color_bar = '#ff6b35' if proba >= 0.32 else '#00ff9d'
            ax.barh([0], [proba], color=color_bar, height=0.5, alpha=0.8, edgecolor='white', linewidth=2)
            ax.axvline(x=0.32, color='#00d4ff', linestyle='--', linewidth=3, label='Seuil optimal (0.32)')
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Probabilité d\'accident', color='white', fontsize=12)
            ax.set_title('Visualisation du risque', color='#00ff9d', fontsize=14, fontweight='bold')
            ax.tick_params(colors='white')
            ax.legend(facecolor='#1a1d3e', edgecolor='#00d4ff', labelcolor='white')
            
            for spine in ax.spines.values():
                spine.set_color('#00d4ff')
                spine.set_linewidth(1.5)
            
            st.pyplot(fig)
            
            if modalite == "Oui":
                st.error(f"⚠️ **ATTENTION : Risque d'accident ÉLEVÉ détecté ({proba*100:.1f}%)**")
                st.warning("Recommandations : Prudence maximale conseillée dans ces conditions!")
            else:
                st.success(f"✅ **Risque d'accident FAIBLE dans ces conditions ({proba*100:.1f}%)**")
                st.info("Restez néanmoins vigilant et respectez le code de la route.")
    
    # SECTION 7: CONCLUSION
    elif section == "📝 Conclusion":
        st.markdown('<h2 class="section-title">7. Conclusion générale</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="conclusion-box">
        <h3>🎯 Synthèse de l'étude</h3>
        
        <h4>📊 Résultats principaux</h4>
        <p>Cette étude statistique a analysé <strong>840 observations</strong> pour identifier les facteurs 
        influençant la survenue d'accidents routiers à partir d'un ensemble de données comprenant 14 variables.</p>
        
        <h4>🔍 Facteurs identifiés</h4>
        <ul>
            <li><strong>✅ Facteur significatif :</strong> L'<strong>état de la route</strong> (p = 0.024) 
            est le seul facteur statistiquement associé à la survenue d'accidents parmi toutes les variables testées</li>
            <li><strong>📈 Impact spécifique :</strong> Les routes <strong>sèches</strong> multiplient par 1.78 
            les chances d'accident par rapport aux routes en construction (OR = 1.78)</li>
            <li><strong>❌ Facteurs non significatifs :</strong> Contrairement aux attentes, météo, heure, 
            consommation d'alcool, type de route, âge et expérience du conducteur n'ont pas montré d'association 
            statistique significative dans cette analyse</li>
        </ul>
        
        <h4>🤖 Performance du modèle de régression logistique</h4>
        <ul>
            <li><strong>Pseudo R²</strong> : 0.033 (capacité prédictive limitée)</li>
            <li><strong>Précision globale</strong> : ~65% (taux de mauvais classement de 35%)</li>
            <li><strong>Seuil optimal</strong> : 0.32 (déterminé par l'indice de Youden)</li>
            <li><strong>Qualité des résidus</strong> : 100% des observations dans les limites acceptables [-3, 3]</li>
            <li><strong>Variables significatives</strong> : Constante et état de la route (sec)</li>
        </ul>
        
        <h4>💡 Implications pratiques pour la prévention routière</h4>
        <ul>
            <li><strong>Surveillance accrue</strong> : Renforcer l'entretien et le contrôle de l'état des routes</li>
            <li><strong>Sensibilisation ciblée</strong> : Alerter sur les risques des routes sèches 
            (excès de confiance, vitesse excessive, sous-estimation du danger)</li>
            <li><strong>Signalisation améliorée</strong> : Mieux identifier et baliser les zones à risque</li>
            <li><strong>Infrastructure</strong> : Investir dans l'amélioration de la qualité des chaussées</li>
        </ul>
        
        <h4>⚠️ Limites identifiées</h4>
        <ul>
            <li><strong>Performance modérée</strong> : 35% d'erreur de classification indique une marge d'amélioration importante</li>
            <li><strong>Facteurs classiques absents</strong> : Certains facteurs reconnus (alcool, météo) 
            n'apparaissent pas comme significatifs, possiblement en raison de la taille de l'échantillon 
            ou de la variabilité des données</li>
            <li><strong>Faible R²</strong> : Le modèle explique seulement 3.3% de la variance, suggérant 
            l'existence de facteurs non capturés</li>
            <li><strong>Données agrégées</strong> : Manque potentiel de granularité dans certaines mesures</li>
        </ul>
        
        <h4>🔬 Perspectives et pistes d'amélioration</h4>
        <p><strong>Variables supplémentaires à intégrer :</strong></p>
        <ul>
            <li><strong>Facteurs humains</strong> : Fatigue du conducteur, utilisation du téléphone, 
            stress, niveau d'attention</li>
            <li><strong>Conditions environnementales détaillées</strong> : Météo en temps réel, 
            visibilité précise, température de la chaussée</li>
            <li><strong>Données géospatiales</strong> : Localisation GPS, topographie, densité urbaine, 
            historique d'accidents par zone</li>
            <li><strong>Infrastructure routière</strong> : Qualité de la signalisation, présence de virages 
            dangereux, état de l'éclairage public, travaux en cours</li>
            <li><strong>Comportements dynamiques</strong> : Vitesse réelle, accélérations/freinages brusques 
            captés par capteurs embarqués</li>
        </ul>
        
        <p><strong>Approches méthodologiques avancées :</strong></p>
        <ul>
            <li><strong>Machine Learning</strong> : Random Forest, XGBoost, Gradient Boosting pour capturer 
            des relations non-linéaires</li>
            <li><strong>Deep Learning</strong> : Réseaux de neurones pour modéliser des interactions complexes</li>
            <li><strong>Analyse d'interactions</strong> : Explorer les effets combinés entre facteurs 
            (ex: météo + état route + heure)</li>
            <li><strong>Segmentation</strong> : Modèles spécifiques par type de route ou de région</li>
            <li><strong>Séries temporelles</strong> : Intégrer l'évolution temporelle des accidents</li>
        </ul>
        
        <h4>🎓 Conclusion finale</h4>
        <p>Bien que le modèle actuel présente des limitations en termes de capacité prédictive (Pseudo R² = 0.033), 
        cette étude apporte une contribution significative à la compréhension des facteurs d'accidents routiers 
        en identifiant l'<strong>état de la route</strong> comme facteur clé significatif.</p>
        
        <p>Les résultats invitent à poursuivre la recherche avec :</p>
        <ul>
            <li>Des échantillons plus larges pour détecter des effets plus subtils</li>
            <li>Des variables explicatives plus diversifiées et granulaires</li>
            <li>Des approches méthodologiques enrichies (machine learning, deep learning)</li>
            <li>Une exploration approfondie des interactions entre facteurs</li>
        </ul>
        
        <p><strong>📈 Impact attendu :</strong> Une meilleure modélisation du risque d'accident pourrait contribuer 
        significativement à la réduction du nombre d'accidents et à l'amélioration de la sécurité routière globale 
        par des politiques de prévention plus ciblées et efficaces.</p>
        
        <p style="text-align: center; margin-top: 30px; font-size: 1.2rem; color: #00ff9d;">
        <strong>La sécurité routière est l'affaire de tous. Chaque accident évité est une vie sauvée.</strong>
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Récapitulatif des métriques clés de l'étude")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Observations", "840")
            st.metric("📈 Variables", "14")
        with col2:
            st.metric("🎯 Taux accidents", f"{(df['Accident']=='Oui').mean()*100:.1f}%")
            st.metric("✅ Facteurs significatifs", "1")
        with col3:
            st.metric("🤖 Pseudo R²", f"{model.prsquared:.3f}")
            st.metric("📊 Précision", "~65%")
        with col4:
            st.metric("🎲 Seuil optimal", "0.32")
            st.metric("❌ Taux erreur", "~35%")
        
        st.markdown("---")
        
        st.markdown("""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(138, 43, 226, 0.1)); 
        border-radius: 15px; border: 2px solid rgba(0, 212, 255, 0.3); margin: 30px 0;'>
        <h3 style='color: #00d4ff; font-family: Orbitron, sans-serif; text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);'>
        🚗 Application d'Analyse Statistique des Accidents Routiers</h3>
        <p style='color: #00ff9d; font-size: 1.1rem; margin-top: 15px;'>
        Analyse et modélisation du risque d'accident par régression logistique</p>
        <p style='color: #c0c0c0; margin-top: 10px;'>
        Développé dans le cadre d'une étude statistique approfondie | 2025</p>
        <p style='color: #8a2be2; margin-top: 15px; font-style: italic;'>
        "Vers une meilleure compréhension des facteurs d'accidents pour une prévention efficace"</p>
        </div>
        """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("""
    ❌ **Erreur : Fichier de données introuvable**
    
    Le fichier `dataset_accident.xlsx` n'a pas été trouvé dans le répertoire courant.
    
    **Vérifiez que :**
    - Le fichier est bien nommé `dataset_accident.xlsx`
    - Il se trouve dans le même dossier que ce script Python
    - Vous avez les droits de lecture sur le fichier
    
    **Chemin actuel de recherche :** Dossier d'exécution du script
    """)
    st.stop()

except Exception as e:
    st.error(f"""
    ❌ **Erreur inattendue lors du traitement**
    
    Une erreur s'est produite lors du chargement ou du traitement des données.
    
    **Détails de l'erreur :**
    ```
    {str(e)}
    ```
    
    **Suggestions :**
    - Vérifiez l'intégrité du fichier Excel
    - Assurez-vous que toutes les colonnes requises sont présentes
    - Vérifiez que les données sont dans le bon format
    """)
    st.stop()