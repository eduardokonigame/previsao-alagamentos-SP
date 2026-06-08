from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import requests
import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sistema Preditivo de Alagamentos - São Paulo")

# Permitir acesso do Lovable (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pega o caminho absoluto da pasta onde este main.py está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Variáveis globais para os modelos reais
xgb = None
rf = None
mlp = None
scaler_mlp = None
le = None
hmm = None
scaler_hmm = None
tabela_bayes = None
nomes_hmm = None

def carregar_componentes():
    global xgb, rf, mlp, scaler_mlp, le, hmm, scaler_hmm, tabela_bayes, nomes_hmm
    try:
        xgb = joblib.load(os.path.join(BASE_DIR, "modelo_xgboost.pkl"))
        rf = joblib.load(os.path.join(BASE_DIR, "modelo_alagamento_rf.pkl"))
        mlp = joblib.load(os.path.join(BASE_DIR, "modelo_mlp.pkl"))
        scaler_mlp = joblib.load(os.path.join(BASE_DIR, "scaler_mlp.pkl"))
        le = joblib.load(os.path.join(BASE_DIR, "label_encoder_subprefeitura.pkl"))
        hmm = joblib.load(os.path.join(BASE_DIR, "modelo_hmm_solo.pkl"))
        scaler_hmm = joblib.load(os.path.join(BASE_DIR, "scaler_hmm.pkl"))
        tabela_bayes = joblib.load(os.path.join(BASE_DIR, "tabela_bayes.pkl"))
        nomes_hmm = joblib.load(os.path.join(BASE_DIR, "nomes_estados_hmm.pkl"))
        
        print("✅ TODOS os componentes reais carregados com sucesso!")
        return True
    except Exception as e:
        print(f"❌ ERRO ao carregar arquivos .pkl: {e}")
        return False

# Inicializa o carregamento dos modelos
componentes_prontos = carregar_componentes()

# Dicionário de Coordenadas Geográficas para busca automática via Satélite
COORDENADAS_SP = {
    "ARICANDUVA": {"lat": -23.5622, "lon": -46.5186}, "BUTANTA": {"lat": -23.5714, "lon": -46.7083},
    "CAMPO LIMPO": {"lat": -23.6335, "lon": -46.7561}, "CAPELA DO SOCORRO": {"lat": -23.6975, "lon": -46.6997},
    "CASA VERDE": {"lat": -23.5042, "lon": -46.6508}, "CIDADE ADEMAR": {"lat": -23.6669, "lon": -46.6575},
    "CIDADE TIRADENTES": {"lat": -23.5936, "lon": -46.4022}, "ERMELINO MATARAZZO": {"lat": -23.4961, "lon": -46.4994},
    "FREGUESIA DO O": {"lat": -23.4906, "lon": -46.6992}, "GUAIANASES": {"lat": -23.5425, "lon": -46.4169},
    "IPIRANGA": {"lat": -23.5902, "lon": -46.6102}, "ITAIM PAULISTA": {"lat": -23.4947, "lon": -46.3986},
    "ITAQUERA": {"lat": -23.5358, "lon": -46.4550}, "JABAQUARA": {"lat": -23.6456, "lon": -46.6439},
    "JAÇANA": {"lat": -23.4561, "lon": -46.5925}, "LAPA": {"lat": -23.5226, "lon": -46.7019},
    "M BOI MIRIM": {"lat": -23.6747, "lon": -46.7628}, "MOOCA": {"lat": -23.5505, "lon": -46.6033},
    "PARELHEIROS": {"lat": -23.8217, "lon": -46.6853}, "PENHA": {"lat": -23.5244, "lon": -46.5451},
    "PERUS": {"lat": -23.4072, "lon": -46.7486}, "PINHEIROS": {"lat": -23.5654, "lon": -46.6994},
    "PIRITUBA": {"lat": -23.4831, "lon": -46.7219}, "SANTANA": {"lat": -23.5028, "lon": -46.6253},
    "SANTO AMARO": {"lat": -23.6492, "lon": -46.7042}, "SÃO MATEUS": {"lat": -23.6139, "lon": -46.4756},
    "SÃO MIGUEL PAULISTA": {"lat": -23.4942, "lon": -46.4422}, "SAPOPEMBA": {"lat": -23.5956, "lon": -46.5108},
    "SE": {"lat": -23.5486, "lon": -46.6341}, "TREMEMBE": {"lat": -23.4439, "lon": -46.6156},
    "VILA MARIANA": {"lat": -23.5891, "lon": -46.6342}, "VILA PRUDENTE": {"lat": -23.5855, "lon": -46.5625}
}

def buscar_clima_completo_open_meteo(lat: float, lon: float):
    """Busca dados climáticos em tempo real e históricos via Open-Meteo API"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=rain&daily=rain_sum&past_days=7&timezone=America/Sao_Paulo"
        response = requests.get(url, timeout=5).json()
        chuva_atual = response.get("current", {}).get("rain", 0.0)
        lista_chuva_diaria = response.get("daily", {}).get("rain_sum", [])
        
        if len(lista_chuva_diaria) < 7:
            return chuva_atual, 0.0, 0.0
            
        chuva_3d = sum(lista_chuva_diaria[-3:])
        chuva_7d = sum(lista_chuva_diaria[-7:]) / 7.0
        return chuva_atual, chuva_3d, chuva_7d
    except Exception as e:
        print(f"Erro ao acessar Open-Meteo: {e}")
        return 0.0, 0.0, 0.0

# Schema flexível para aceitar os dois modos (Manual e Automático)
class DadosEntrada(BaseModel):
    modo: str = "manual"  # "manual" ou "automatico"
    subprefeitura: str
    chuva: float = 0.0
    chuva_acumulada_3d: float = 0.0
    chuva_media_7d: float = 0.0
    mes: int = 6
    dia_semana: int = 0

@app.post("/predict")
async def predict(dados: DadosEntrada):
    if not componentes_prontos:
        raise HTTPException(status_code=500, detail="Os modelos lógicos não foram carregados com sucesso no servidor.")
    
    try:
        # 1. Validar e Traduzir Subprefeitura
        nome_sub = dados.subprefeitura.upper().strip()
        if nome_sub not in le.classes_:
            raise HTTPException(status_code=400, detail=f"Subprefeitura '{nome_sub}' não cadastrada no LabelEncoder.")
        sub_id = le.transform([nome_sub])[0]
        
        # 2. Definição do Modo de Entrada (Híbrido)
        modo = dados.modo.lower().strip()
        coords = COORDENADAS_SP.get(nome_sub, {"lat": -23.5505, "lon": -46.6033})
        
        if modo == "automatico":
            chuva_final, chuva_3d, chuva_7d = buscar_clima_completo_open_meteo(coords["lat"], coords["lon"])
            hoje = datetime.datetime.now()
            mes_final = hoje.month
            dia_semana_final = hoje.weekday()
            origem_dados = "Sensores Satélite (Open-Meteo API)"
        else:
            chuva_final = dados.chuva
            chuva_3d = dados.chuva_acumulada_3d
            chuva_7d = dados.chuva_media_7d
            mes_final = dados.mes
            dia_semana_final = dados.dia_semana
            origem_dados = "Inserção Manual (Simulação de Cenário)"
        
        # 3. Processamento nos Modelos Classificadores Reais
        features = [[chuva_final, chuva_3d, chuva_7d, mes_final, dia_semana_final, sub_id]]
        df_input = pd.DataFrame(features, columns=['chuva', 'chuva_acumulada_3d', 'chuva_media_7d', 'mes', 'dia_semana', 'sub_id'])
        
        prob_xgb = float(xgb.predict_proba(df_input)[:, 1][0]) * 100
        prob_rf = float(rf.predict_proba(df_input)[:, 1][0]) * 100
        
        X_scaled = scaler_mlp.transform(df_input)
        prob_mlp = float(mlp.predict_proba(X_scaled)[:, 1][0]) * 100
        
        # 4. Processamento no Modelo Temporal de Dinâmica do Solo (HMM)
        X_hmm = scaler_hmm.transform([[chuva_3d, chuva_7d]])
        estado_id = hmm.predict(X_hmm)[0]
        estado_nome = nomes_hmm[estado_id]
        
        # 5. Cálculo do Consenso Consolidado (Média dos Classificadores)
        risco_medio = round((prob_xgb + prob_rf + prob_mlp) / 3, 1)
        consenso_label = "ALTO" if risco_medio > 60 else "MÉDIO" if risco_medio > 30 else "BAIXO"
        
        return {
            "status": "sucesso",
            "modo_executado": modo,
            "analise_solo_hmm": estado_nome.upper(),  # MODELO 1: HMM
            "predicoes_modelos": {
                "random_forest": round(prob_rf, 1),   # MODELO 2: RF
                "xgboost": round(prob_xgb, 1),       # MODELO 3: XGBoost
                "mlp": round(prob_mlp, 1)            # MODELO 4: MLP Network
            },
            "consenso_final": consenso_label,
            "risco_percentual_medio": risco_medio,
            "dados_clima": {
                "chuva_1h": round(chuva_final, 2),
                "acumulado_3d": round(chuva_3d, 2),
                "media_7d": round(chuva_7d, 2),
                "origem": origem_dados
            },
            "laudo": f"Modo: {origem_dados}. O modelo HMM classificou a dinâmica do solo como {estado_nome.upper()}. O consenso preditivo calculado via média aritmética simples entre os algoritmos (Random Forest: {prob_rf:.1f}%, XGBoost: {prob_xgb:.1f}%, MLP: {prob_mlp:.1f}%) indica risco médio consolidated de alagamento em {risco_medio}%."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
