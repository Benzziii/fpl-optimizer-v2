"""
Mesin Prediksi Expected Points (xP) Multi-Variabel & Multi-Horizon (1 GW, 5 GW, 12 GW)
Mencakup:
1. Harga pemain & efisiensi value
2. Status pemain (Fit, Doubtful, Injured, Suspended) & Chance to Play (0-100%)
3. Set-Piece Taker (Penalty order, Direct Free Kick order, Corner Kick order)
4. Akumulasi kartu kuning (risiko suspensi 5 YC sebelum GW19 & 10 YC sebelum GW32) + kartu merah
5. Statistik Head-to-Head (H2H) historis antar tim & faktor Kandang/Tandang (Home/Away)
6. Statistik 10-20 pertandingan terakhir untuk pemain (xG/90, xA/90, xMin, BPS) dan tim (xGF, xGA)
"""
import math
from typing import Dict, List, Any

GOAL_POINTS = {1: 6, 2: 6, 3: 5, 4: 4}  # GK: 6, DEF: 6, MID: 5, FWD: 4
CS_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
ASSIST_POINTS = 3.0


def poisson_clean_sheet_prob(expected_goals_against: float) -> float:
    """Peluang Clean Sheet P(X = 0) dari distribusi Poisson dengan parameter lambda = xGA."""
    return math.exp(-max(0.15, expected_goals_against))


def compute_yellow_card_suspension_factor(
    yellow_cards: int,
    current_gw: int,
    horizon_offset: int,
    yc_per_90: float = 0.18
) -> float:
    """
    Menghitung faktor pengurang menit bermain akibat risiko akumulasi kartu kuning:
    - Paruh 1 (s.d. GW19): 5 kartu kuning = larangan 1 pertandingan
    - Paruh 2 (s.d. GW32): 10 kartu kuning = larangan 2 pertandingan
    """
    target_gw = current_gw + horizon_offset
    projected_yc = yellow_cards + (horizon_offset * yc_per_90)

    if target_gw <= 19:
        if yellow_cards == 4:
            # Pemain sudah mengantongi 4 YC: risiko terkena kartu ke-5 dan absen di GW berikutnya
            if horizon_offset == 0:
                return 0.94  # bermain hati-hati agar tidak terkena kartu
            elif horizon_offset == 1:
                return 1.0 - min(0.45, yc_per_90 * 1.6)  # peluang absen karena akumulasi
            else:
                return 0.92
        elif projected_yc >= 4.5:
            return 0.90
    elif target_gw <= 32:
        if yellow_cards in (8, 9):
            # Risiko suspensi 2 pertandingan jika mencapai 10 YC
            ban_prob = 0.35 if yellow_cards == 9 else 0.20
            return 1.0 - (ban_prob if horizon_offset >= 1 else 0.05)
    return 1.0


def compute_set_piece_boost(
    penalties_order: int,
    direct_fk_order: int,
    corners_order: int,
    element_type: int,
    team_attack_index: float = 1.0
) -> float:
    """
    Menghitung tambahan Expected Points (xP) dari tugas eksekutor bola mati (Set-Piece Taker).
    - Penalty Taker Utama (#1): ~0.22 penalti/laga * 0.79 konversi * poin gol
    - Free Kick Taker Utama (#1): tambahan xG langsung & rebound assist
    - Corner & Indirect FK Taker (#1): tambahan xA dari umpan silang bola mati
    """
    goal_pts = GOAL_POINTS.get(element_type, 4)
    boost = 0.0

    # Penalti
    if penalties_order == 1:
        boost += 0.175 * goal_pts * team_attack_index
    elif penalties_order == 2:
        boost += 0.035 * goal_pts * team_attack_index

    # Tendangan Bebas Langsung (Direct Free Kick)
    if direct_fk_order == 1:
        boost += (0.045 * goal_pts + 0.03 * ASSIST_POINTS) * team_attack_index
    elif direct_fk_order == 2:
        boost += 0.018 * goal_pts * team_attack_index

    # Sepak Pojok (Corner Kick)
    if corners_order == 1:
        boost += 0.095 * ASSIST_POINTS * team_attack_index
    elif corners_order == 2:
        boost += 0.045 * ASSIST_POINTS * team_attack_index

    return round(boost, 3)


def predict_player_xp_for_fixture(
    player: Dict[str, Any],
    fixture_info: Dict[str, Any],
    current_gw: int,
    horizon_offset: int,
    weights: Dict[str, float]
) -> float:
    """
    Menghitung prediksi poin (xP) seorang pemain pada satu pertandingan tertentu
    berdasarkan seluruh variabel optimasi.
    """
    pos = player["element_type"]  # 1=GK, 2=DEF, 3=MID, 4=FWD
    status = player.get("status", "a")
    chance_this_round = player.get("chance_of_playing", 100)

    # 1. Faktor Status & Chance to Play (pulih bertahap untuk horizon > 1 GW)
    if status in ("u", "s") and horizon_offset == 0:
        avail_factor = 0.0
    elif status == "i":
        if chance_this_round == 0:
            avail_factor = 0.0 if horizon_offset == 0 else min(0.85, 0.25 * horizon_offset)
        else:
            avail_factor = (chance_this_round / 100.0) if horizon_offset == 0 else min(0.95, (chance_this_round / 100.0) + 0.15 * horizon_offset)
    elif status == "d":
        avail_factor = (chance_this_round / 100.0) if horizon_offset == 0 else min(1.0, (chance_this_round / 100.0) + 0.20 * horizon_offset)
    else:
        avail_factor = 1.0

    if avail_factor <= 0.01:
        return 0.0

    # 2. Ekspektasi Menit Bermain (xMin) dari 10-20 laga terakhir & Akumulasi Kartu
    avg_mins_last_15 = player.get("avg_minutes_last_15", 82.0)
    yc_factor = compute_yellow_card_suspension_factor(
        yellow_cards=player.get("yellow_cards", 0),
        current_gw=current_gw,
        horizon_offset=horizon_offset,
        yc_per_90=player.get("yc_per_90", 0.16)
    )
    yc_factor = 1.0 - (1.0 - yc_factor) * weights.get("card_risk_weight", 1.0)
    expected_mins = avg_mins_last_15 * avail_factor * yc_factor
    mins_ratio = min(1.0, expected_mins / 90.0)

    # Poin penampilan (Appearance Points: >=60 menit = 2 poin, <60 menit = 1 poin)
    appearance_xp = 2.0 * min(1.0, expected_mins / 65.0) if expected_mins >= 45 else (expected_mins / 45.0)

    # 3. Faktor Kandang/Tandang (Home/Away) & Rekor Head-to-Head (H2H)
    is_home = fixture_info.get("is_home", True)
    home_away_mult = 1.08 if is_home else 0.93
    h2h_attack_mult = fixture_info.get("h2h_attack_mult", 1.0)  # >1.0 jika tim dominan secara H2H historis
    h2h_def_mult = fixture_info.get("h2h_def_mult", 1.0)        # <1.0 jika lawan sering gagal mencetak gol di H2H

    combined_matchup_attack = (
        1.0 + ((home_away_mult * h2h_attack_mult) - 1.0) * weights.get("h2h_venue_weight", 1.0)
    )

    # 4. Statistik 10-20 Pertandingan Terakhir (Pemain & Tim)
    xg_90_last_15 = player.get("xg_per_90_last_15", 0.25)
    xa_90_last_15 = player.get("xa_per_90_last_15", 0.18)
    team_att_last_15 = player.get("team_attack_form_15", 1.05)
    opp_def_weakness_15 = fixture_info.get("opp_xga_index_15", 1.0)

    # Ekspektasi Gol & Assist Terbobot
    attack_scale = (
        weights.get("player_form_15_weight", 1.0) * 0.65
        + weights.get("team_form_15_weight", 1.0) * 0.35 * team_att_last_15
    ) * opp_def_weakness_15 * combined_matchup_attack

    exp_goals = xg_90_last_15 * mins_ratio * attack_scale
    exp_assists = xa_90_last_15 * mins_ratio * attack_scale

    attacking_xp = exp_goals * GOAL_POINTS[pos] + exp_assists * ASSIST_POINTS

    # 5. Tambahan Set-Piece Taker (PK, FK, CK)
    sp_boost = compute_set_piece_boost(
        penalties_order=player.get("penalties_order", 0),
        direct_fk_order=player.get("direct_fk_order", 0),
        corners_order=player.get("corners_order", 0),
        element_type=pos,
        team_attack_index=team_att_last_15 * combined_matchup_attack
    ) * mins_ratio * weights.get("set_piece_weight", 1.0)

    # 6. Ekspektasi Clean Sheet, Saves (GK), & Kebobolan (-1 per 2 gol utk GK/DEF)
    team_xga_15 = player.get("team_xga_last_15", 1.15)
    opp_att_15 = fixture_info.get("opp_xgf_index_15", 1.0)
    venue_def_adj = 0.90 if is_home else 1.12
    expected_goals_conceded = team_xga_15 * opp_att_15 * venue_def_adj * h2h_def_mult

    cs_prob = poisson_clean_sheet_prob(expected_goals_conceded)
    cs_xp = cs_prob * CS_POINTS[pos] * min(1.0, expected_mins / 65.0)

    goals_conceded_penalty = 0.0
    if pos in (1, 2):
        goals_conceded_penalty = -0.45 * expected_goals_conceded * mins_ratio

    saves_xp = 0.0
    if pos == 1:
        saves_xp = player.get("saves_per_90", 3.1) * 0.33 * mins_ratio

    # 7. Ekspektasi Bonus Points (BPS) & Penalti Kartu
    bps_xp = player.get("bonus_per_90_last_15", 0.45) * mins_ratio * combined_matchup_attack
    card_deduction = -1.0 * player.get("yc_per_90", 0.16) * mins_ratio

    # Diskon waktu (Time Decay) untuk horizon panjang (5 GW / 12 GW)
    decay = (weights.get("horizon_decay", 0.95)) ** horizon_offset

    raw_xp = (
        appearance_xp
        + attacking_xp
        + sp_boost
        + cs_xp
        + saves_xp
        + bps_xp
        + goals_conceded_penalty
        + card_deduction
    )
    return round(max(0.0, raw_xp * avail_factor * decay), 2)
