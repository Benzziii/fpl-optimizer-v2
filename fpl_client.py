"""
Modul Pengambil Data Resmi Fantasy Premier League (FPL API)
Mendukung pengambilan data pemain, jadwal (fixtures), histori pertandingan,
dan data tim manajer berdasarkan FPL Manager ID.
"""
import requests
from typing import Dict, Any, List, Optional

FPL_BASE_URL = "https://fantasy.premierleague.com/api"

class FPLDataClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) FPL-Quantum-Optimizer/2.0"
        })

    def get_bootstrap_static(self) -> Dict[str, Any]:
        """Mengambil data seluruh pemain, tim, aturan gameweek, dan set-piece notes."""
        resp = self.session.get(f"{FPL_BASE_URL}/bootstrap-static/", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_fixtures(self) -> List[Dict[str, Any]]:
        """Mengambil seluruh jadwal pertandingan musim berjalan (termasuk skor & statistik H2H musim ini)."""
        resp = self.session.get(f"{FPL_BASE_URL}/fixtures/", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_player_summary(self, element_id: int) -> Dict[str, Any]:
        """Mengambil histori 10-20 pertandingan terakhir pemain & jadwal ke depan."""
        resp = self.session.get(f"{FPL_BASE_URL}/element-summary/{element_id}/", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_manager_info(self, manager_id: int) -> Dict[str, Any]:
        """Mengambil profil manajer, saldo bank, dan informasi tim."""
        resp = self.session.get(f"{FPL_BASE_URL}/entry/{manager_id}/", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_manager_history(self, manager_id: int) -> Dict[str, Any]:
        """Mengambil histori GW manajer dan daftar chip yang sudah terpakai."""
        resp = self.session.get(f"{FPL_BASE_URL}/entry/{manager_id}/history/", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_manager_picks(self, manager_id: int, event_gw: int) -> Dict[str, Any]:
        """Mengambil 15 pemain dalam skuad manajer pada Gameweek tertentu."""
        resp = self.session.get(f"{FPL_BASE_URL}/entry/{manager_id}/event/{event_gw}/picks/", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def compute_free_transfers(self, history_data: Dict[str, Any], chips_used: List[Dict[str, Any]]) -> int:
        """
        Menghitung sisa Free Transfer (FT) secara akurat (mendukung aturan akumulasi hingga 5 FT).
        Wildcard dan Free Hit tidak menghabiskan akumulasi FT sesuai regulasi terbaru FPL.
        """
        current_gws = history_data.get("current", [])
        if not current_gws:
            return 1
        chip_map = {c["event"]: c["name"] for c in chips_used}
        ft = 1
        for gw_row in current_gws:
            ev = gw_row["event"]
            transfers_made = gw_row.get("event_transfers", 0)
            cost = gw_row.get("event_transfers_cost", 0)
            chip = chip_map.get(ev)
            if chip in ("wildcard", "freehit"):
                ft = min(5, ft)
            else:
                paid_transfers = cost // 4
                free_used = max(0, transfers_made - paid_transfers)
                ft = min(5, max(1, ft - free_used + 1))
        return ft
