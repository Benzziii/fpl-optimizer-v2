"""
Server API FastAPI untuk Aplikasi Web Optimasi & Prediksi FPL
Menyediakan endpoint bebas kendala CORS untuk mengambil data Manager ID langsung dari FPL API
serta menjalankan solver MILP PuLP untuk horizon 1 GW, 5 GW, hingga 12 GW.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from fpl_client import FPLDataClient
from xp_predictor import predict_player_xp_for_fixture
from optimizer_milp import evaluate_all_transfer_and_chip_strategies

app = FastAPI(title="FPL Quantum Optimizer & Multi-Horizon MILP API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = FPLDataClient()


class OptimizeRequest(BaseModel):
    players: List[Dict[str, Any]]
    current_squad_ids: List[int]
    total_budget: float
    free_transfers: int = 1
    horizon_gws: int = 5
    available_chips: List[str] = ["wildcard", "freehit", "benchboost", "triplecaptain"]


@app.get("/api/fpl/manager/{manager_id}")
def get_fpl_manager_bundle(manager_id: int, gw: Optional[int] = None):
    """Mengambil data lengkap profil manajer, histori, sisa FT, chip, dan 15 pemain saat ini."""
    try:
        info = client.get_manager_info(manager_id)
        history = client.get_manager_history(manager_id)
        current_event = gw or info.get("current_event") or 1
        picks = client.get_manager_picks(manager_id, current_event)
        chips_used = history.get("chips", [])
        computed_ft = client.compute_free_transfers(history, chips_used)
        return {
            "manager_id": manager_id,
            "manager_name": f"{info.get('player_first_name', '')} {info.get('player_last_name', '')}".strip(),
            "team_name": info.get("name", "FPL Squad"),
            "current_event": current_event,
            "bank_m": (picks.get("entry_history", {}).get("bank", 0)) / 10.0,
            "value_m": (picks.get("entry_history", {}).get("value", 1000)) / 10.0,
            "free_transfers": computed_ft,
            "chips_used": chips_used,
            "picks": picks.get("picks", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil data FPL ID {manager_id}: {exc}")


@app.post("/api/optimize")
def run_milp_optimization(req: OptimizeRequest):
    """Menjalankan optimasi MILP (Roll vs Free Transfer vs Hit vs Chip) untuk 1/5/12 GW."""
    return evaluate_all_transfer_and_chip_strategies(
        players=req.players,
        current_squad_ids=req.current_squad_ids,
        total_budget=req.total_budget,
        free_transfers=req.free_transfers,
        horizon_gws=req.horizon_gws,
        available_chips=req.available_chips,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
