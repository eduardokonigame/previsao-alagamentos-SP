from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Permitir acesso do Lovable (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pega o caminho absoluto da pasta onde este main.py está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Variáveis globais para os modelos
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
        # Carregamento usando caminhos absolutos para evitar erros no Mac
        xgb = joblib.load(os.path.join(BASE_DIR, "modelo_xgboost.pkl"))
        rf = joblib.load(os.path.join(BASE_DIR, "modelo_alagamento_rf.pkl"))
        mlp = joblib.load(os.path.join(BASE_DIR, "modelo_mlp.pkl"))
        scaler_mlp = joblib.load(os.path.join(BASE_DIR, "scaler_mlp.pkl"))
        le = joblib.load(os.path.join(BASE_DIR, "label_encoder_subprefeitura.pkl"))
        hmm = joblib.load(os.path.join(BASE_DIR, "modelo_hmm_solo.pkl"))
        scaler_hmm = joblib.load(os.path.join(BASE_DIR, "scaler_hmm.pkl"))
        tabela_bayes = joblib.load(os.path.join(BASE_DIR, "tabela_bayes.pkl"))
        nomes_hmm = joblib.load(os.path.join(BASE_DIR, "nomes_estados_hmm.pkl"))
        
        print("✅ TODOS os componentes carregados com sucesso!")
        return True
    except Exception as e:
        print(f"❌ ERRO ao carregar arquivos: {e}")
        return False

# Inicializa o carregamento
componentes_prontos = carregar_componentes()

class DadosEntrada(BaseModel):
    chuva: float
    chuva_acumulada_3d: float
    chuva_media_7d: float
    mes: int
    dia_semana: int
    subprefeitura: str

@app.post("/prever")
async def prever(dados: DadosEntrada):
    if not componentes_prontos:
        raise HTTPException(status_code=500, detail="Os modelos não foram carregados no servidor.")
    
    try:
        # 1. Traduzir Subprefeitura
        nome_sub = dados.subprefeitura.upper().strip()
        if nome_sub not in le.classes_:
            raise HTTPException(status_code=400, detail=f"Subprefeitura '{nome_sub}' não reconhecida.")
            
        sub_id = le.transform([nome_sub])[0]
        
        # 2. Preparar dados
        features = [[dados.chuva, dados.chuva_acumulada_3d, dados.chuva_media_7d, 
                     dados.mes, dados.dia_semana, sub_id]]
        df_input = pd.DataFrame(features, columns=['chuva', 'chuva_acumulada_3d', 'chuva_media_7d', 'mes', 'dia_semana', 'sub_id'])

        # 3. Predições (Probabilidades)
        prob_xgb = float(xgb.predict_proba(df_input)[:, 1][0])
        prob_rf = float(rf.predict_proba(df_input)[:, 1][0])
        X_scaled = scaler_mlp.transform(df_input)
        prob_mlp = float(mlp.predict_proba(X_scaled)[:, 1][0])

        # 4. Explicabilidade (HMM)
        X_hmm = scaler_hmm.transform([[dados.chuva_acumulada_3d, dados.chuva_media_7d]])
        estado_id = hmm.predict(X_hmm)[0]
        estado_nome = nomes_hmm[estado_id]
        vulnerabilidade = tabela_bayes[estado_id]

        # 5. Consenso Final
        media_prob = (prob_xgb + prob_rf + prob_mlp) / 3
        consenso = "ALTO" if media_prob > 0.6 else "MÉDIO" if media_prob > 0.3 else "BAIXO"

        return {
            "status": "sucesso",
            "consenso_final": consenso,
            "risco_percentual": round(media_prob * 100, 1),
            "analise_solo": estado_nome,
            "laudo": f"O solo na região está {estado_nome}. O consenso dos modelos indica risco {consenso} ({media_prob*100:.1f}%)."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # No Render, a porta será lida automaticamente, localmente usamos 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)