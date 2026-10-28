"""
Model Optimasi Mixed-Integer Linear Programming (MILP) Multi-Horizon FPL
Menggunakan library PuLP (CBC Solver) untuk menyelesaikan:
- Pemilihan 15 Pemain (2 GK, 5 DEF, 5 MID, 3 FWD), Maks 3 pemain per klub, Batasan Budget
- Pemilihan Starting XI (1 GK, min 3 DEF, min 2 MID, min 1 FWD)
- Pemilihan Kapten (2x atau 3x saat Triple Captain) & Wakil Kapten
- Evaluasi Strategi Transfer:
  1) Roll Transfer (0 transfer, simpan FT untuk GW depan)
  2) Gunakan Free Transfer tanpa potongan poin (Cost = 0)
  3) Multi-Transfer dengan Potongan Poin (Hit -4 poin per ekstra transfer) jika Net xP lebih tinggi
- Optimasi Penggunaan 4 Chip (2x kesempatan di Paruh 1 [GW1-19] & Paruh 2 [GW20-38]):
  Triple Captain, Bench Boost, Free Hit, dan Wildcard
- Mendukung Horizon 1 GW, 5 GW, hingga 12 GW ke depan.
"""
import pulp
from typing import List, Dict, Any, Optional


def solve_fpl_horizon_milp(
    players: List[Dict[str, Any]],
    current_squad_ids: List[int],
    total_budget: float,
    free_transfers: int,
    horizon_gws: int = 5,
    max_transfers_allowed: Optional[int] = None,
    active_chip: Optional[str] = None,  # None, "wildcard", "freehit", "benchboost", "triplecaptain"
    locked_ids: Optional[List[int]] = None,
    excluded_ids: Optional[List[int]] = None,
    roll_ft_bonus: float = 1.65
) -> Dict[str, Any]:
    """
    Menyelesaikan masalah optimasi skuad FPL untuk skenario transfer/chip tertentu
    dan mengembalikan skuad optimal, Starting XI, Kapten, serta Net Expected Points (xP).
    """
    locked_ids = set(locked_ids or [])
    excluded_ids = set(excluded_ids or [])
    current_set = set(current_squad_ids)

    prob = pulp.LpProblem(f"FPL_MultiHorizon_{horizon_gws}GW", pulp.LpMaximize)

    p_ids = [p["id"] for p in players]
    p_map = {p["id"]: p for p in players}

    # Variabel Keputusan:
    # x[i] = 1 jika pemain i masuk dalam 15 pemain skuad baru
    x = pulp.LpVariable.dicts("squad", p_ids, cat=pulp.LpBinary)
    # s[i] = 1 jika pemain i menjadi Starter (Starting XI) pada GW berjalan
    s = pulp.LpVariable.dicts("start", p_ids, cat=pulp.LpBinary)
    # c[i] = 1 jika pemain i dipilih sebagai Kapten
    c = pulp.LpVariable.dicts("captain", p_ids, cat=pulp.LpBinary)
    # vc[i] = 1 jika pemain i dipilih sebagai Wakil Kapten
    vc = pulp.LpVariable.dicts("vice_captain", p_ids, cat=pulp.LpBinary)

    # Variabel jumlah transfer keluar/masuk dan jumlah Hit (-4 poin)
    transfers_in = pulp.LpSum([x[i] for i in p_ids if i not in current_set])
    paid_hits = pulp.LpVariable("paid_hits", lowBound=0, cat=pulp.LpInteger)

    # 1. Batasan Struktur Skuad 15 Pemain & Budget
    prob += pulp.LpSum([x[i] for i in p_ids]) == 15, "Total_15_Players"
    prob += pulp.LpSum([x[i] * p_map[i]["price"] for i in p_ids]) <= total_budget + 1e-4, "Budget_Cap"

    # Komposisi Posisi Skuad (2 GK, 5 DEF, 5 MID, 3 FWD)
    for pos, req_count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        prob += pulp.LpSum([x[i] for i in p_ids if p_map[i]["element_type"] == pos]) == req_count, f"Squad_Pos_{pos}"

    # Maksimal 3 pemain dari 1 tim Premier League
    teams = set(p["team"] for p in players)
    for t in teams:
        prob += pulp.LpSum([x[i] for i in p_ids if p_map[i]["team"] == t]) <= 3, f"Max_3_Team_{t}"

    # 2. Batasan Starting XI (11 Pemain) & Formasi Valid
    prob += pulp.LpSum([s[i] for i in p_ids]) == 11, "Starting_11"
    for i in p_ids:
        prob += s[i] <= x[i], f"Start_In_Squad_{i}"
        prob += c[i] <= s[i], f"Cap_In_Start_{i}"
        prob += vc[i] <= s[i], f"VC_In_Start_{i}"
        prob += c[i] + vc[i] <= 1, f"Distinct_Cap_VC_{i}"

    prob += pulp.LpSum([c[i] for i in p_ids]) == 1, "One_Captain"
    prob += pulp.LpSum([vc[i] for i in p_ids]) == 1, "One_ViceCaptain"

    # Aturan Formasi Valid FPL: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD
    prob += pulp.LpSum([s[i] for i in p_ids if p_map[i]["element_type"] == 1]) == 1, "Start_1_GK"
    prob += pulp.LpSum([s[i] for i in p_ids if p_map[i]["element_type"] == 2]) >= 3, "Start_Min_3_DEF"
    prob += pulp.LpSum([s[i] for i in p_ids if p_map[i]["element_type"] == 3]) >= 2, "Start_Min_2_MID"
    prob += pulp.LpSum([s[i] for i in p_ids if p_map[i]["element_type"] == 4]) >= 1, "Start_Min_1_FWD"

    # Kunci / Eksklusi Pemain
    for lid in locked_ids:
        if lid in x:
            prob += x[lid] == 1, f"Lock_{lid}"
    for eid in excluded_ids:
        if eid in x:
            prob += x[eid] == 0, f"Exclude_{eid}"

    # 3. Aturan Transfer & Potongan Poin (Hit Cost)
    if max_transfers_allowed is not None:
        prob += transfers_in <= max_transfers_allowed, "Max_Transfers_Cap"

    if active_chip in ("wildcard", "freehit"):
        prob += paid_hits == 0, "No_Hit_On_WC_FH"
    else:
        prob += paid_hits >= transfers_in - free_transfers, "Paid_Hits_Lower_Bound"

    # 4. Fungsi Objektif Multi-Horizon
    # GW berjalan (GW 1 dari horizon): Starter mendapat 100% xP, Kapten +100% (atau +200% jika Triple Captain),
    # Bench mendapat bobot kontribusi (100% jika Bench Boost, atau ~12% auto-sub probability jika normal).
    # GW 2..H dari horizon: Jika Free Hit aktif, hanya dievaluasi untuk 1 GW lonjakan + skuad lama kembali di GW 2..H.
    bench_weight_gw1 = 1.0 if active_chip == "benchboost" else 0.12
    cap_mult_gw1 = 2.0 if active_chip == "triplecaptain" else 1.0

    gw1_xp_expr = pulp.LpSum([
        s[i] * p_map[i]["xp_gw_list"][0]
        + c[i] * cap_mult_gw1 * p_map[i]["xp_gw_list"][0]
        + vc[i] * 0.08 * p_map[i]["xp_gw_list"][0]
        + (x[i] - s[i]) * bench_weight_gw1 * p_map[i]["xp_gw_list"][0]
        for i in p_ids
    ])

    future_horizon_xp_expr = 0
    if horizon_gws > 1 and active_chip != "freehit":
        # Untuk GW ke-2 s.d. ke-H, kontribusi pemain dihitung dengan bobot rotasi optimal (Starter utama ~0.82, kapten terbaik)
        future_horizon_xp_expr = pulp.LpSum([
            x[i] * sum(p_map[i]["xp_gw_list"][1:horizon_gws]) * 0.78
            + c[i] * sum(p_map[i]["xp_gw_list"][1:horizon_gws]) * 0.65
            for i in p_ids
        ])

    hit_penalty_expr = 4.0 * paid_hits

    prob += gw1_xp_expr + future_horizon_xp_expr - hit_penalty_expr, "Total_Net_Horizon_xP"

    # Jalankan solver CBC tanpa log berlebih
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    selected_squad = [p_map[i] for i in p_ids if pulp.value(x[i]) > 0.5]
    starters = [p_map[i] for i in p_ids if pulp.value(s[i]) > 0.5]
    bench = [p_map[i] for i in p_ids if pulp.value(x[i]) > 0.5 and pulp.value(s[i]) <= 0.5]
    captain = next((p_map[i] for i in p_ids if pulp.value(c[i]) > 0.5), starters[0] if starters else None)
    vice_captain = next((p_map[i] for i in p_ids if pulp.value(vc[i]) > 0.5), starters[1] if len(starters) > 1 else None)

    transfers_out_players = [p_map[i] for i in current_squad_ids if i in p_map and pulp.value(x[i]) <= 0.5]
    transfers_in_players = [p_map[i] for i in p_ids if i not in current_set and pulp.value(x[i]) > 0.5]

    num_transfers = len(transfers_in_players)
    hits_taken = int(round(pulp.value(paid_hits) or 0))
    hit_cost = hits_taken * 4

    gw1_gross = sum(p["xp_gw_list"][0] for p in starters) + (
        (2.0 if active_chip == "triplecaptain" else 1.0) * (captain["xp_gw_list"][0] if captain else 0.0)
    ) + (sum(p["xp_gw_list"][0] for p in bench) if active_chip == "benchboost" else sum(p["xp_gw_list"][0] for p in bench) * 0.12)

    horizon_gross = gw1_gross
    if horizon_gws > 1:
        for h in range(1, horizon_gws):
            gw_h_sorted = sorted(selected_squad, key=lambda item: item["xp_gw_list"][h], reverse=True)
            top11_h = gw_h_sorted[:11]
            cap_h = top11_h[0] if top11_h else None
            horizon_gross += sum(p["xp_gw_list"][h] for p in top11_h) + (cap_h["xp_gw_list"][h] if cap_h else 0.0)

    unused_ft_bonus = roll_ft_bonus if (num_transfers == 0 and active_chip is None and free_transfers < 5) else 0.0
    horizon_net = round(horizon_gross - hit_cost + unused_ft_bonus, 2)

    return {
        "status": pulp.LpStatus[prob.status],
        "active_chip": active_chip,
        "horizon_gws": horizon_gws,
        "num_transfers": num_transfers,
        "hits_taken": hits_taken,
        "hit_cost": hit_cost,
        "unused_ft_bonus": round(unused_ft_bonus, 2),
        "gw1_gross_xp": round(gw1_gross, 2),
        "horizon_gross_xp": round(horizon_gross, 2),
        "horizon_net_xp": horizon_net,
        "total_squad_cost": round(sum(p["price"] for p in selected_squad), 1),
        "captain": captain,
        "vice_captain": vice_captain,
        "starters": starters,
        "bench": bench,
        "transfers_out": transfers_out_players,
        "transfers_in": transfers_in_players,
    }


def evaluate_all_transfer_and_chip_strategies(
    players: List[Dict[str, Any]],
    current_squad_ids: List[int],
    total_budget: float,
    free_transfers: int,
    horizon_gws: int,
    available_chips: List[str]
) -> Dict[str, Any]:
    """
    Membandingkan secara menyeluruh:
    1. Roll Transfer (0 Transfer - Simpan FT)
    2. Optimal Free Transfer (Tanpa Hit)
    3. Optimal Hit Strategy (1 s.d. 4 Hit ekstra selama Net ROI lebih tinggi)
    4. Strategi Aktivasi Chip (Triple Captain, Bench Boost, Free Hit, Wildcard)
    """
    # 1. Roll Transfer (0 Transfer)
    roll_res = solve_fpl_horizon_milp(
        players=players,
        current_squad_ids=current_squad_ids,
        total_budget=total_budget,
        free_transfers=free_transfers,
        horizon_gws=horizon_gws,
        max_transfers_allowed=0
    )
    roll_res["strategy_label"] = "Simpan Transfer (Roll FT)"

    # 2. Gunakan Free Transfer Saja (0 Hit)
    ft_res = solve_fpl_horizon_milp(
        players=players,
        current_squad_ids=current_squad_ids,
        total_budget=total_budget,
        free_transfers=free_transfers,
        horizon_gws=horizon_gws,
        max_transfers_allowed=free_transfers
    )
    ft_res["strategy_label"] = f"Gunakan Free Transfer ({ft_res['num_transfers']} FT, 0 Hit)"

    # 3. Evaluasi Multi-Transfer dengan Potongan Poin (Hit -4, -8, -12)
    hit_res = solve_fpl_horizon_milp(
        players=players,
        current_squad_ids=current_squad_ids,
        total_budget=total_budget,
        free_transfers=free_transfers,
        horizon_gws=horizon_gws,
        max_transfers_allowed=min(6, free_transfers + 3)
    )
    hit_res["strategy_label"] = f"Agresif / Hit ({hit_res['num_transfers']} Transfer, -{hit_res['hit_cost']} pts)"

    # 4. Evaluasi Chip yang Tersedia
    chip_results = {}
    for chip in available_chips:
        c_res = solve_fpl_horizon_milp(
            players=players,
            current_squad_ids=current_squad_ids,
            total_budget=total_budget,
            free_transfers=free_transfers,
            horizon_gws=horizon_gws,
            active_chip=chip
        )
        chip_results[chip] = c_res

    strategies = [roll_res, ft_res, hit_res]
    best_standard = max(strategies, key=lambda item: item["horizon_net_xp"])

    return {
        "best_standard_strategy": best_standard,
        "roll_strategy": roll_res,
        "free_transfer_strategy": ft_res,
        "hit_strategy": hit_res,
        "chip_strategies": chip_results,
    }
