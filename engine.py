"""
FASE 2 & 3: Motor Matemático — Centro de Control de Predicción Deportiva
Modelos Poisson, Fuerza de Ataque/Defensa, Micromercados, Value Betting.
"""

import sqlite3
import os
import numpy as np
import pandas as pd
from scipy.stats import poisson

try:
    from tactical_scout import load_team_tactics, load_squad_players
except Exception:
    load_team_tactics = lambda t: None
    load_squad_players = lambda t: []

DB_PATH = os.path.join(os.path.dirname(__file__), "sports_control_center.db")

# Mapa de código de liga → nombre display
LEAGUE_DISPLAY = {
    "SP1": "LaLiga Primera",
    "SP2": "LaLiga Segunda (Hypermotion)",
    "E0":  "Premier League",
    "I1":  "Serie A",
    "D1":  "Bundesliga",
    "F1":  "Ligue 1",
    "P1":  "Primeira Liga",
}

# Mapa inverso
DISPLAY_TO_CODE = {v: k for k, v in LEAGUE_DISPLAY.items()}

# Partidos sugeridos para la semana por liga (Fase 3)
WEEKLY_FIXTURES = {
    "SP1": [
        ("Real Madrid", "Barcelona"),
        ("Atletico Madrid", "Sevilla"),
        ("Athletic Club", "Villarreal"),
        ("Valencia", "Real Betis"),
    ],
    "SP2": [
        ("Espanyol", "Racing Santander"),
        ("Levante", "Real Valladolid"),
    ],
    "E0": [
        ("Arsenal", "Chelsea"),
        ("Manchester City", "Liverpool"),
        ("Tottenham", "Manchester Utd"),
        ("Newcastle", "Aston Villa"),
    ],
    "I1": [
        ("Juventus", "Inter"),
        ("Napoli", "AC Milan"),
        ("Lazio", "Roma"),
    ],
    "D1": [
        ("Bayern Munich", "Dortmund"),
        ("Leverkusen", "Leipzig"),
        ("Hoffenheim", "Freiburg"),
    ],
    "F1": [
        ("Paris SG", "Marseille"),
        ("Lyon", "Monaco"),
        ("Lens", "Lille"),
    ],
    "P1": [
        ("Porto", "Benfica"),
        ("Sporting CP", "Braga"),
        ("Vitoria SC", "Guimaraes"),
    ],
}


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def load_matches(div: str = None) -> pd.DataFrame:
    """Carga partidos desde SQLite, opcionalmente filtrando por liga."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB no encontrada: {DB_PATH}. Ejecuta ingest.py primero.")
    conn = sqlite3.connect(DB_PATH)
    if div:
        df = pd.read_sql(
            "SELECT * FROM matches WHERE Div = ?", conn, params=(div,)
        )
    else:
        df = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["FTHG", "FTAG", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR",
                "B365H", "B365D", "B365A"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_understat_xg(div: str = None) -> pd.DataFrame:
    """
    Carga la tabla understat_xg desde SQLite.
    Retorna DataFrame con columnas: team, xg, xga, div.
    Devuelve DataFrame vacío si la tabla no existe (graceful degradation).
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["team", "xg", "xga", "div"])
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT team, xg, xga, div FROM understat_xg"
        if div:
            query += " WHERE div = ?"
            df = pd.read_sql(query, conn, params=(div,))
        else:
            df = pd.read_sql(query, conn)
        conn.close()
        for col in ["xg", "xga"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=["team", "xg", "xga", "div"])


def get_teams(div: str) -> list[str]:
    """Retorna lista ordenada de equipos de una liga."""
    df = load_matches(div)
    teams = sorted(set(df["HomeTeam"].dropna().tolist() + df["AwayTeam"].dropna().tolist()))
    return teams


def get_available_leagues() -> list[str]:
    """Retorna las ligas disponibles en la DB (en formato display)."""
    if not os.path.exists(DB_PATH):
        return list(LEAGUE_DISPLAY.values())
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT DISTINCT Div FROM matches", conn)
    conn.close()
    codes = df["Div"].tolist()
    return [LEAGUE_DISPLAY.get(c, c) for c in codes if c in LEAGUE_DISPLAY]


# =============================================================================
# FASE 2.1 — TIME DECAY, ELO RATING Y FUERZA LOCAL / VISITANTE
# =============================================================================

def compute_decay_weights(df: pd.DataFrame, half_life_days: float = 40.0) -> np.ndarray:
    """
    Calcula pesos de decaimiento exponencial temporal para cada partido.
    w_i = exp(-xi * Delta_t) con xi = ln(2) / half_life_days.
    Otorga a los partidos recientes (ultimas 5-6 semanas) un impacto
    significativamente mayor que a los partidos de hace meses.
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        if dates.notna().sum() > 0:
            max_date = dates.max()
            days_ago = (max_date - dates).dt.total_seconds() / (24 * 3600)
            median_days = days_ago.median() if not np.isnan(days_ago.median()) else 30.0
            days_ago = days_ago.fillna(median_days).clip(lower=0)
            xi = np.log(2) / max(1.0, float(half_life_days))
            weights = np.exp(-xi * days_ago).to_numpy(dtype=float)
            return np.maximum(weights, 1e-4)

    # Fallback si no hay fechas válidas: decaimiento exponencial por orden de fila
    ranks = np.arange(n)
    weights = np.exp(-0.06 * (n - 1 - ranks))
    return np.maximum(weights, 1e-4)


def compute_elo_ratings(df: pd.DataFrame, base_elo: float = 1500.0, k_factor: float = 20.0, home_adv: float = 65.0) -> dict:
    """
    Calcula ratings Elo (Club Elo standard) para cada equipo.
    K=20, HFA=65 puntos. Sin factor de margen de victoria (Club Elo puro).
    """
    all_teams = set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna())
    ratings = {team: float(base_elo) for team in all_teams}

    sorted_df = df.sort_values("Date") if "Date" in df.columns else df
    valid_matches = sorted_df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    for _, row in valid_matches.iterrows():
        h = str(row["HomeTeam"]).strip()
        a = str(row["AwayTeam"]).strip()
        if h not in ratings or a not in ratings:
            continue
        try:
            hg = float(row["FTHG"])
            ag = float(row["FTAG"])
        except (ValueError, TypeError):
            continue

        if hg > ag:
            s_h, s_a = 1.0, 0.0
        elif hg == ag:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        r_h = ratings[h]
        r_a = ratings[a]

        # Esperanza Elo con ventaja de campo (Club Elo standard)
        e_h = 1.0 / (1.0 + 10.0 ** (-((r_h + home_adv) - r_a) / 400.0))
        e_a = 1.0 - e_h

        ratings[h] += k_factor * (s_h - e_h)
        ratings[a] += k_factor * (s_a - e_a)

    return ratings


def compute_team_strengths(df: pd.DataFrame, half_life_days: float = 40.0) -> dict:
    """
    Calcula índices de fuerza de ataque y defensa para cada equipo con separación
    estricta Local / Visitante:
    - Fuerza Local: calculada filtrando sus partidos en CASA (goles a favor vs media local, en contra vs visitante).
    - Fuerza Visitante: calculada filtrando sus partidos FUERA (goles a favor vs media visitante, en contra vs local).
    Aplica ponderación temporal exponencial (Time Decay) en cada split.
    Retorna: { equipo: {att_home, def_home, att_away, def_away} }
    """
    weights = compute_decay_weights(df, half_life_days=half_life_days)

    fthg_valid = ~df["FTHG"].isna()
    ftag_valid = ~df["FTAG"].isna()

    w_home = weights[fthg_valid]
    w_away = weights[ftag_valid]

    league_avg_home = (
        float(np.average(df.loc[fthg_valid, "FTHG"], weights=w_home))
        if len(w_home) > 0 and np.sum(w_home) > 0
        else 1.3
    )
    league_avg_away = (
        float(np.average(df.loc[ftag_valid, "FTAG"], weights=w_away))
        if len(w_away) > 0 and np.sum(w_away) > 0
        else 1.1
    )

    if league_avg_home <= 0 or np.isnan(league_avg_home):
        league_avg_home = 1.3
    if league_avg_away <= 0 or np.isnan(league_avg_away):
        league_avg_away = 1.1

    strengths = {}
    all_teams = set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna())

    for team in all_teams:
        home_mask = (df["HomeTeam"] == team) & fthg_valid & ftag_valid
        away_mask = (df["AwayTeam"] == team) & fthg_valid & ftag_valid

        home_w = weights[home_mask]
        away_w = weights[away_mask]

        # Fallback equilibrado por si el equipo tiene muy pocos partidos en un split
        overall_mask = ((df["HomeTeam"] == team) | (df["AwayTeam"] == team)) & fthg_valid & ftag_valid
        overall_w = weights[overall_mask]
        if np.sum(overall_mask) > 0 and np.sum(overall_w) > 0:
            goals_for = np.where(df.loc[overall_mask, "HomeTeam"] == team,
                                 df.loc[overall_mask, "FTHG"],
                                 df.loc[overall_mask, "FTAG"])
            goals_against = np.where(df.loc[overall_mask, "HomeTeam"] == team,
                                     df.loc[overall_mask, "FTAG"],
                                     df.loc[overall_mask, "FTHG"])
            mid_avg = (league_avg_home + league_avg_away) / 2.0
            fallback_att = float(np.average(goals_for, weights=overall_w) / mid_avg)
            fallback_def = float(np.average(goals_against, weights=overall_w) / mid_avg)
        else:
            fallback_att, fallback_def = 1.0, 1.0

        # Fuerza estricta como local (CASA) con Time Decay
        if np.sum(home_mask) >= 3 and np.sum(home_w) > 0:
            att_h = float(np.average(df.loc[home_mask, "FTHG"], weights=home_w) / league_avg_home)
            def_h = float(np.average(df.loc[home_mask, "FTAG"], weights=home_w) / league_avg_away)
        else:
            att_h, def_h = fallback_att, fallback_def

        # Fuerza estricta como visitante (FUERA) con Time Decay
        if np.sum(away_mask) >= 3 and np.sum(away_w) > 0:
            att_a = float(np.average(df.loc[away_mask, "FTAG"], weights=away_w) / league_avg_away)
            def_a = float(np.average(df.loc[away_mask, "FTHG"], weights=away_w) / league_avg_home)
        else:
            att_a, def_a = fallback_att, fallback_def

        strengths[team] = {
            "att_home": max(0.15, float(att_h)),
            "def_home": max(0.15, float(def_h)),
            "att_away": max(0.15, float(att_a)),
            "def_away": max(0.15, float(def_a)),
        }

    return strengths


# =============================================================================
# FASE 2.2 — PONDERACIÓN POR RACHA (MOMENTUM CON SPLIT)
# =============================================================================

def apply_momentum(df: pd.DataFrame, team: str, is_home: bool,
                   base_strength: dict, n_recent: int = 5,
                   weight_recent: float = 0.35) -> dict:
    """
    Pondera la fuerza base con los últimos n partidos del equipo,
    filtrando prioritariamente por condición (CASA para local, FUERA para visitante)
    con decaimiento intra-ventana.
    """
    if team not in base_strength:
        return {"att": 1.0, "def": 1.0}

    league_avg_home = df["FTHG"].mean() if df["FTHG"].mean() > 0 else 1.3
    league_avg_away = df["FTAG"].mean() if df["FTAG"].mean() > 0 else 1.1

    # Filtrar según la condición específica del partido
    if is_home:
        recent_matches = df[df["HomeTeam"] == team].sort_values("Date").tail(n_recent)
        if len(recent_matches) < 3:
            h_m = df[df["HomeTeam"] == team]
            a_m = df[df["AwayTeam"] == team]
            recent_matches = pd.concat([h_m, a_m]).sort_values("Date").tail(n_recent)
    else:
        recent_matches = df[df["AwayTeam"] == team].sort_values("Date").tail(n_recent)
        if len(recent_matches) < 3:
            h_m = df[df["HomeTeam"] == team]
            a_m = df[df["AwayTeam"] == team]
            recent_matches = pd.concat([h_m, a_m]).sort_values("Date").tail(n_recent)

    s = base_strength[team]
    base_att = s["att_home"] if is_home else s["att_away"]
    base_def = s["def_home"] if is_home else s["def_away"]

    if len(recent_matches) == 0:
        return {"att": base_att, "def": base_def}

    recent_att_list, recent_def_list = [], []
    for _, row in recent_matches.iterrows():
        if row["HomeTeam"] == team:
            recent_att_list.append(row["FTHG"] / league_avg_home if league_avg_home else 1.0)
            recent_def_list.append(row["FTAG"] / league_avg_away if league_avg_away else 1.0)
        else:
            recent_att_list.append(row["FTAG"] / league_avg_away if league_avg_away else 1.0)
            recent_def_list.append(row["FTHG"] / league_avg_home if league_avg_home else 1.0)

    k = len(recent_att_list)
    intra_weights = np.exp(np.linspace(-0.5, 0.0, k))
    recent_att = float(np.average(recent_att_list, weights=intra_weights))
    recent_def = float(np.average(recent_def_list, weights=intra_weights))

    final_att = (1.0 - weight_recent) * base_att + weight_recent * recent_att
    final_def = (1.0 - weight_recent) * base_def + weight_recent * recent_def

    return {"att": max(0.1, final_att), "def": max(0.1, final_def)}


# =============================================================================
# FASE 2.3 — SIMULADOR POISSON CON MINI-ELO Y DIXON-COLES
# =============================================================================

def apply_squad_absence(team_name: str, base_xg: float,
                        absent_players: list = None,
                        key_absence_checked: bool = False,
                        team_elo: float = 1500.0) -> tuple[float, list[str]]:
    """
    Penalización quirúrgica por bajas conectada a squad_players y al Elo:
    - Si se selecciona un jugador real de la plantilla, resta su contribución neta
      (jugador_xg90 - suplente_xg90) * (minutos_esperados / 90) modulada por la profundidad
      de plantilla del club:
      * Gigante (Real Madrid/Barcelona, Elo ~1800+): el suplente absorbe el 80-85% de la producción.
      * Tabla baja (Elo ~1400): el suplente solo absorbe el 35-40%.
    - Si solo se marca el checkbox general 'Baja clave', aplica una deducción modulada por Elo.
    """
    squad = load_squad_players(team_name)
    squad_depth = float(np.clip((team_elo - 1350.0) / (1850.0 - 1350.0), 0.0, 1.0))
    backup_absorption = 0.35 + 0.50 * squad_depth  # 0.35 (tabla baja) a 0.85 (gigante)
    unabsorbed_ratio = 1.0 - backup_absorption     # 0.65 (tabla baja) a 0.15 (gigante)

    applied_notes = []
    adjusted_xg = base_xg

    if absent_players:
        for absent in absent_players:
            matched = next((p for p in squad if p["player_name"] == absent or p["player_id"] == absent or absent.startswith(p["player_name"])), None)
            if matched:
                p_xg = matched.get("xg_per_90", 0.40)
                b_xg = p_xg * 0.45
                mins = matched.get("minutes_played", 1800)
                mins_factor = min(1.0, mins / 1800.0)
                net_drop_raw = (p_xg - b_xg) * mins_factor
                net_loss_xg = net_drop_raw * unabsorbed_ratio
                adjusted_xg = max(0.1, adjusted_xg - net_loss_xg)
                applied_notes.append(f"{matched['player_name']} (-{net_loss_xg:.2f} xG, {backup_absorption*100:.0f}% absorción)")
            else:
                loss = adjusted_xg * 0.10 * unabsorbed_ratio
                adjusted_xg = max(0.1, adjusted_xg - loss)
                applied_notes.append(f"{absent} (-{loss:.2f} xG)")

    elif key_absence_checked:
        general_loss_pct = 0.08 + 0.10 * (1.0 - squad_depth)
        adjusted_xg *= (1.0 - general_loss_pct)
        applied_notes.append(f"Baja Clave Genérica (-{general_loss_pct*100:.0f}% xG)")

    return round(adjusted_xg, 3), applied_notes


def apply_tactical_matchup(home_team: str, away_team: str, xg_home: float, xg_away: float) -> dict:
    """
    Cruza los estilos tácticos de ambos equipos antes de Dixon-Coles y Montecarlo:
    - Si Local = 'Dominador Posesión' (>56% pos) y Visitante = 'Bloque Bajo / Contragolpe' (o 'Bloque Bajo'):
      reduce el ritmo goleador total un 6% (partido cerrado, sube Under 2.5 y 0-0/1-0),
      e incrementa la proyección de Córners del local en +1.4.
    - Si ambos equipos son 'Vertical / Transición' o 'Presión Alta Intensa':
      incrementa ambas lambdas (xg_home, xg_away) un 8% por partido de ida y vuelta
      (sube BTTS y Over 2.5) e incrementa Tarjetas Esperadas.
    """
    t_h = load_team_tactics(home_team) or {}
    t_a = load_team_tactics(away_team) or {}

    pos_h = t_h.get("possession_pct", 50.0)
    pos_a = t_a.get("possession_pct", 50.0)
    style_h = t_h.get("tactical_style", "Vertical / Transición")
    style_a = t_a.get("tactical_style", "Vertical / Transición")
    ppda_h = t_h.get("opp_passes_per_def_action", 11.5)
    ppda_a = t_a.get("opp_passes_per_def_action", 11.5)

    corner_boost_home = 0.0
    cards_boost = 0.0
    synergy_note = ""

    is_dominant_home = (style_h == "Dominador Posesión" or pos_h >= 56.0)
    is_low_block_away = ("Bloque Bajo" in style_a)

    if is_dominant_home and is_low_block_away:
        xg_home *= 0.94
        xg_away *= 0.94
        corner_boost_home = 1.4
        synergy_note = "Asedio Local vs Bloque Bajo (-6% ritmo de gol, +1.4 Córners local)"

    fast_styles = ("Vertical / Transición", "Presión Alta Intensa")
    if style_h in fast_styles and style_a in fast_styles:
        xg_home *= 1.08
        xg_away *= 1.08
        cards_boost = 0.6
        synergy_note = "Transición Rápida / Presión Alta (+8% ritmo ida y vuelta, +Tarjetas)"

    return {
        "xg_home": xg_home,
        "xg_away": xg_away,
        "corner_boost_home": corner_boost_home,
        "cards_boost": cards_boost,
        "style_home": style_h,
        "style_away": style_a,
        "possession_home": pos_h,
        "possession_away": pos_a,
        "ppda_home": ppda_h,
        "ppda_away": ppda_a,
        "synergy_note": synergy_note,
    }


def compute_xg(df: pd.DataFrame, home_team: str, away_team: str,
               injury_home: bool = False, injury_away: bool = False,
               injury_factor: float = 0.85,
               absent_players_home: list = None,
               absent_players_away: list = None) -> dict:
    """
    Calcula xG esperados usando distribución de Poisson con:
    - True xG de Understat como base ofensiva/defensiva (si disponible).
      Fallback a goles reales cuando no hay datos Understat.
    - Split estricto Local / Visitante con Time Decay
    - Momentum (últimos partidos)
    - Factor de ajuste de nivel por rating Elo (Mini-Elo, Club Elo K=20/HFA=65)
    - Ajuste por lesión de jugador clave
    """
    # --- Detectar si tenemos datos Understat para esta liga ---
    div_code = df["Div"].iloc[0] if "Div" in df.columns and len(df) > 0 else None
    xg_df = load_understat_xg(div_code) if div_code else pd.DataFrame(columns=["team", "xg", "xga"])

    def _get_understat(team: str):
        """Devuelve (xg_avg, xga_avg) de Understat, o (None, None) si no hay datos."""
        if xg_df.empty:
            return None, None
        row = xg_df[xg_df["team"] == team]
        if row.empty:
            return None, None
        return float(row["xg"].iloc[0]), float(row["xga"].iloc[0])

    # --- Fuerza de ataque/defensa basada en goles reales (con Time Decay) ---
    strengths = compute_team_strengths(df)
    elo_ratings = compute_elo_ratings(df)

    home_s = apply_momentum(df, home_team, is_home=True, base_strength=strengths)
    away_s = apply_momentum(df, away_team, is_home=False, base_strength=strengths)

    # Medias de goles reales de la liga (referencia para escalar)
    league_avg_home = df["FTHG"].mean() if df["FTHG"].mean() > 0 else 1.3
    league_avg_away = df["FTAG"].mean() if df["FTAG"].mean() > 0 else 1.1

    # --- Understat True xG ---
    h_xg, h_xga = _get_understat(home_team)
    a_xg, a_xga = _get_understat(away_team)

    has_understat_home = h_xg is not None and a_xga is not None
    has_understat_away = a_xg is not None and h_xga is not None

    if has_understat_home and has_understat_away:
        # Lambda basada en True xG: xG ofensivo × xGA defensivo / referencia de liga
        league_ref_xg = xg_df["xg"].mean() if not xg_df.empty else league_avg_home
        league_ref_xga = xg_df["xga"].mean() if not xg_df.empty else league_avg_away
        league_ref_xg  = max(league_ref_xg, 0.5)
        league_ref_xga = max(league_ref_xga, 0.5)

        xg_home_base = (h_xg / league_ref_xg) * (a_xga / league_ref_xga) * league_ref_xg
        xg_away_base = (a_xg / league_ref_xg) * (h_xga / league_ref_xga) * league_ref_xga

        # Blend 60% True xG + 40% modelo goles reales (robustez)
        xg_home_goals = home_s["att"] * away_s["def"] * league_avg_home
        xg_away_goals = away_s["att"] * home_s["def"] * league_avg_away
        xg_home = 0.6 * xg_home_base + 0.4 * xg_home_goals
        xg_away = 0.6 * xg_away_base + 0.4 * xg_away_goals
        source = "understat+goals"
    else:
        # Fallback completo a goles reales
        xg_home = home_s["att"] * away_s["def"] * league_avg_home
        xg_away = away_s["att"] * home_s["def"] * league_avg_away
        source = "goals_only"

    # --- Factor de ajuste por nivel relativo (Mini-Elo) ---
    r_h = elo_ratings.get(home_team, 1500.0)
    r_a = elo_ratings.get(away_team, 1500.0)
    elo_diff = r_h - r_a

    elo_adj_home = float(np.clip(np.exp(elo_diff / 1000.0), 0.70, 1.45))
    elo_adj_away = float(np.clip(np.exp(-elo_diff / 1000.0), 0.70, 1.45))

    xg_home *= elo_adj_home
    xg_away *= elo_adj_away

    # --- Ajuste quirúrgico por bajas / profundidad de plantilla ---
    xg_home, home_abs_notes = apply_squad_absence(
        home_team, xg_home,
        absent_players=absent_players_home,
        key_absence_checked=injury_home,
        team_elo=r_h
    )
    xg_away, away_abs_notes = apply_squad_absence(
        away_team, xg_away,
        absent_players=absent_players_away,
        key_absence_checked=injury_away,
        team_elo=r_a
    )

    return {
        "xg_home": max(0.1, round(xg_home, 3)),
        "xg_away": max(0.1, round(xg_away, 3)),
        "home_elo": round(r_h, 1),
        "away_elo": round(r_a, 1),
        "elo_diff": round(elo_diff, 1),
        "home_abs_notes": home_abs_notes,
        "away_abs_notes": away_abs_notes,
        # True xG de Understat para mostrar en UI (None si no disponibles)
        "true_xg_home": round(h_xg, 3) if h_xg is not None else None,
        "true_xga_home": round(h_xga, 3) if h_xga is not None else None,
        "true_xg_away": round(a_xg, 3) if a_xg is not None else None,
        "true_xga_away": round(a_xga, 3) if a_xga is not None else None,
        "xg_source": source,
    }


def dixon_coles_tau(x: int, y: int, xg_home: float, xg_away: float, rho: float = -0.15) -> float:
    """
    Función de ajuste Tau (τ) de Dixon-Coles (1997) para corregir el sesgo
    de independencia de Poisson en marcadores de baja puntuación:
    Aplica el ajuste EXCLUSIVAMENTE a las celdas (0,0), (1,0), (0,1) y (1,1).
    - (0,0): 1.0 - lambda * mu * rho
    - (0,1): 1.0 + lambda * rho
    - (1,0): 1.0 + mu * rho
    - (1,1): 1.0 - rho
    - Otros marcadores: 1.0 (neutro)
    """
    if x == 0 and y == 0:
        val = 1.0 - (xg_home * xg_away * rho)
    elif x == 0 and y == 1:
        val = 1.0 + (xg_home * rho)
    elif x == 1 and y == 0:
        val = 1.0 + (xg_away * rho)
    elif x == 1 and y == 1:
        val = 1.0 - rho
    else:
        val = 1.0
    return max(0.001, float(val))


def build_score_matrix(xg_home: float, xg_away: float, max_goals: int = 7, rho: float = -0.15) -> np.ndarray:
    """
    Construye la matriz bivariada de probabilidades de goles [home_goals x away_goals]
    aplicando la corrección matemática de Dixon-Coles EXCLUSIVAMENTE a los marcadores
    (0,0), (1,0), (0,1) y (1,1), y normalizando para que la suma total sea exactamente 1.0.
    """
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_ind = poisson.pmf(i, xg_home) * poisson.pmf(j, xg_away)
            if (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                tau = dixon_coles_tau(i, j, xg_home, xg_away, rho=rho)
                matrix[i, j] = p_ind * tau
            else:
                matrix[i, j] = p_ind

    total_p = np.sum(matrix)
    if total_p > 0:
        matrix /= total_p
    return matrix


def compute_1x2_probs(matrix: np.ndarray) -> dict:
    """Calcula probabilidades 1X2 desde la matriz de marcadores normalizada."""
    p_home = float(np.sum(np.tril(matrix, -1)))   # home goals > away goals
    p_draw = float(np.trace(matrix))               # diagonal
    p_away = float(np.sum(np.triu(matrix, 1)))     # away goals > home goals
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
    return {"home": round(p_home, 4), "draw": round(p_draw, 4), "away": round(p_away, 4)}


def simulate_match_montecarlo(lambda_home: float, lambda_away: float, iteraciones: int = 10000) -> dict:
    """
    Simula el partido mediante 10.000 iteraciones de Montecarlo por fuerza bruta
    usando numpy.random.poisson sobre las lambdas (xG ajustados) de cada equipo.
    Calcula dinámicamente las probabilidades puras de Over 2.5 y BTTS por conteo directo de eventos.
    """
    lh = max(0.05, float(lambda_home))
    la = max(0.05, float(lambda_away))

    sim_home = np.random.poisson(lam=lh, size=iteraciones)
    sim_away = np.random.poisson(lam=la, size=iteraciones)

    total_goals = sim_home + sim_away
    over_2_5 = float(np.mean(total_goals > 2))
    under_2_5 = 1.0 - over_2_5

    btts_yes = float(np.mean((sim_home > 0) & (sim_away > 0)))
    btts_no = 1.0 - btts_yes

    return {
        "over_2_5": round(over_2_5, 4),
        "under_2_5": round(under_2_5, 4),
        "btts_yes": round(btts_yes, 4),
        "btts_no": round(btts_no, 4),
        "montecarlo_sims": iteraciones,
    }


def compute_over_btts(matrix_or_home: any = None, lambda_away: float = None, iteraciones: int = 10000) -> dict:
    """
    Calcula Over/Under 2.5 y BTTS mediante el motor de simulación de Montecarlo (10.000 iteraciones).
    Sustituye la aproximación estática de Poisson.
    """
    if isinstance(matrix_or_home, (int, float)) and lambda_away is not None:
        return simulate_match_montecarlo(matrix_or_home, lambda_away, iteraciones)
    elif isinstance(matrix_or_home, np.ndarray):
        xg_h = float(np.sum(matrix_or_home * np.arange(matrix_or_home.shape[0])[:, np.newaxis]))
        xg_a = float(np.sum(matrix_or_home * np.arange(matrix_or_home.shape[1])[np.newaxis, :]))
        return simulate_match_montecarlo(xg_h, xg_a, iteraciones)
    return simulate_match_montecarlo(1.3, 1.1, iteraciones)


def top_exact_scores(matrix: np.ndarray, top_n: int = 3) -> list[tuple]:
    """Retorna los N marcadores exactos más probables según la matriz ajustada."""
    scores = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            scores.append((i, j, matrix[i, j]))
    scores.sort(key=lambda x: x[2], reverse=True)
    return [(f"{s[0]}-{s[1]}", round(s[2] * 100, 2)) for s in scores[:top_n]]


def full_match_prediction(df: pd.DataFrame, home_team: str, away_team: str,
                           injury_home: bool = False, injury_away: bool = False,
                           rho: float = -0.15, sims: int = 10000,
                           absent_players_home: list = None,
                           absent_players_away: list = None) -> dict:
    """Predicción completa de un partido (xG ajustado, sinergia táctica, 1X2 con Dixon-Coles, Over/Under y BTTS con Montecarlo 10.000 sims)."""
    xg = compute_xg(df, home_team, away_team,
                    injury_home=injury_home, injury_away=injury_away,
                    absent_players_home=absent_players_home,
                    absent_players_away=absent_players_away)

    # Matriz de Sinergia Táctica (cruza estilos antes de Dixon-Coles y Montecarlo)
    tac = apply_tactical_matchup(home_team, away_team, xg["xg_home"], xg["xg_away"])
    xg_home_adj = max(0.1, round(tac["xg_home"], 3))
    xg_away_adj = max(0.1, round(tac["xg_away"], 3))

    matrix = build_score_matrix(xg_home_adj, xg_away_adj, rho=rho)
    probs_1x2 = compute_1x2_probs(matrix)
    over_btts = simulate_match_montecarlo(xg_home_adj, xg_away_adj, iteraciones=sims)
    top_scores = top_exact_scores(matrix)

    return {
        **xg,
        "xg_home": xg_home_adj,
        "xg_away": xg_away_adj,
        "tactical_matchup": tac,
        **probs_1x2,
        **over_btts,
        "top_scores": top_scores,
    }


# =============================================================================
# FASE 2.4 — MICROMERCADOS
# =============================================================================

def compute_micromarkets(df: pd.DataFrame, home_team: str, away_team: str) -> dict:
    """
    Calcula líneas esperadas y probabilidades Poisson enriquecidas con métricas tácticas:
    - Córners totales (modulados por posesión territorial y sinergia asedio vs bloque bajo)
    - Tarjetas amarillas (moduladas por PPDA e intensidad de transición)
    - Tiros a puerta (alimentados con shots_per_90 de team_tactics)
    """
    result = {}
    t_h = load_team_tactics(home_team) or {}
    t_a = load_team_tactics(away_team) or {}

    pos_h = t_h.get("possession_pct", 50.0)
    pos_a = t_a.get("possession_pct", 50.0)
    shots_h_90 = t_h.get("shots_per_90", 12.0)
    shots_a_90 = t_a.get("shots_per_90", 11.0)
    style_h = t_h.get("tactical_style", "")
    style_a = t_a.get("tactical_style", "")

    # --- CÓRNERS ---
    if "HC" in df.columns and "AC" in df.columns:
        hc_home = df[df["HomeTeam"] == home_team]["HC"].dropna()
        ac_home = df[df["HomeTeam"] == home_team]["AC"].dropna()
        hc_away = df[df["AwayTeam"] == away_team]["HC"].dropna()
        ac_away = df[df["AwayTeam"] == away_team]["AC"].dropna()

        avg_home_corners = hc_home.mean() if len(hc_home) > 2 else df["HC"].mean()
        avg_away_corners = ac_away.mean() if len(ac_away) > 2 else df["AC"].mean()

        # Ajuste cruzado (defensivo)
        opp_hc = df[df["AwayTeam"] == away_team]["HC"].mean()
        opp_ac = df[df["HomeTeam"] == home_team]["AC"].mean()

        exp_home_c = (avg_home_corners + (opp_hc if not np.isnan(opp_hc) else avg_home_corners)) / 2
        exp_away_c = (avg_away_corners + (opp_ac if not np.isnan(opp_ac) else avg_away_corners)) / 2

        # Modulación táctica: Dominador Local vs Bloque Bajo (+1.4 córners al local)
        if (style_h == "Dominador Posesión" or pos_h >= 56.0) and "Bloque Bajo" in style_a:
            exp_home_c += 1.4

        # Ponderación por posesión relativa
        pos_ratio = pos_h / max(30.0, pos_a)
        exp_home_c *= (0.85 + 0.15 * pos_ratio)
        exp_away_c *= (0.85 + 0.15 * (1.0 / pos_ratio))
        exp_total_c = exp_home_c + exp_away_c

        result["corners_expected_home"] = round(exp_home_c, 2)
        result["corners_expected_away"] = round(exp_away_c, 2)
        result["corners_expected_total"] = round(exp_total_c, 2)

        # Probabilidad Over 8.5 y 9.5 usando Poisson
        result["corners_over_8_5"] = round(1 - poisson.cdf(8, exp_total_c), 4)
        result["corners_over_9_5"] = round(1 - poisson.cdf(9, exp_total_c), 4)
    else:
        result["corners_expected_home"] = None
        result["corners_expected_away"] = None
        result["corners_expected_total"] = None
        result["corners_over_8_5"] = None
        result["corners_over_9_5"] = None

    # --- TARJETAS AMARILLAS ---
    if "HY" in df.columns and "AY" in df.columns:
        hy_home = df[df["HomeTeam"] == home_team]["HY"].dropna()
        ay_home = df[df["HomeTeam"] == home_team]["AY"].dropna()
        hy_away = df[df["AwayTeam"] == away_team]["HY"].dropna()
        ay_away = df[df["AwayTeam"] == away_team]["AY"].dropna()

        avg_hy = hy_home.mean() if len(hy_home) > 2 else df["HY"].mean()
        avg_ay = ay_away.mean() if len(ay_away) > 2 else df["AY"].mean()
        exp_total_cards = (avg_hy if not np.isnan(avg_hy) else 1.5) + \
                          (avg_ay if not np.isnan(avg_ay) else 1.5)

        # Modulación táctica: Ida y vuelta o presión alta (+0.6 tarjetas esperadas)
        fast_styles = ("Vertical / Transición", "Presión Alta Intensa")
        if style_h in fast_styles and style_a in fast_styles:
            exp_total_cards += 0.6

        result["cards_expected_home"] = round(avg_hy if not np.isnan(avg_hy) else 1.5, 2)
        result["cards_expected_away"] = round(avg_ay if not np.isnan(avg_ay) else 1.5, 2)
        result["cards_expected_total"] = round(exp_total_cards, 2)
        result["cards_over_4_5"] = round(1 - poisson.cdf(4, exp_total_cards), 4)
    else:
        result["cards_expected_home"] = None
        result["cards_expected_away"] = None
        result["cards_expected_total"] = None
        result["cards_over_4_5"] = None

    # --- TIROS A PUERTA (Alimentados con shots_per_90 de team_tactics) ---
    if shots_h_90 and shots_a_90:
        st_home = (shots_h_90 * 0.33) * (0.85 + 0.15 * (pos_h / 50.0))
        st_away = (shots_a_90 * 0.33) * (0.85 + 0.15 * (pos_a / 50.0))
        result["shots_expected_home"] = round(st_home, 2)
        result["shots_expected_away"] = round(st_away, 2)
    elif "HST" in df.columns and "AST" in df.columns:
        hst = df[df["HomeTeam"] == home_team]["HST"].dropna()
        ast_ = df[df["AwayTeam"] == away_team]["AST"].dropna()
        league_hst = df["HST"].mean()
        league_ast = df["AST"].mean()

        result["shots_expected_home"] = round(hst.mean() if len(hst) > 2 else league_hst, 2)
        result["shots_expected_away"] = round(ast_.mean() if len(ast_) > 2 else league_ast, 2)
    else:
        result["shots_expected_home"] = None
        result["shots_expected_away"] = None

    return result


# =============================================================================
# FASE 2.5 — ESCÁNER VALUE BETTING (EV+)
# =============================================================================

def compute_fair_odds(prob: float) -> float:
    """Calcula cuota justa sin margen de casa."""
    if prob <= 0:
        return 999.0
    return round(1 / prob, 3)


def compute_ev(prob_model: float, bookie_odds: float) -> float:
    """
    EV = (Prob_Modelo * Cuota_Casa) - 1
    EV > 0 → apuesta con valor esperado positivo.
    """
    if bookie_odds <= 1 or prob_model <= 0:
        return -1.0
    return round((prob_model * bookie_odds) - 1, 4)


def compute_fractional_kelly(prob_model: float, bookie_odds: float, fraction: float = 0.25) -> float:
    """
    Calcula el porcentaje de banca recomendado mediante Criterio de Kelly Fraccionado:
    Fórmula: ((Cuota * Prob_Modelo - 1) / (Cuota - 1)) * fraction
    Retorna 0.0 si EV <= 0 o cuota <= 1.
    """
    if bookie_odds <= 1.0 or prob_model <= 0.0:
        return 0.0
    ev = (prob_model * bookie_odds) - 1.0
    if ev <= 0.0:
        return 0.0
    kelly = ((bookie_odds * prob_model - 1.0) / (bookie_odds - 1.0)) * fraction
    return max(0.0, float(kelly))


def calculate_ev_and_kelly(prob_model: float, bookie_odds: float, fraction: float = 0.25) -> dict:
    """
    Calcula de forma unificada el EV y el Stake Kelly (Quarter Kelly) para un mercado.
    Retorna diccionario seguro con métricas numéricas y formateadas en string.
    """
    ev = compute_ev(prob_model, bookie_odds)
    kelly = compute_fractional_kelly(prob_model, bookie_odds, fraction=fraction)
    return {
        "ev": ev,
        "ev_pct": f"{ev * 100:.1f}%",
        "kelly": kelly,
        "kelly_pct": f"{kelly * 100:.1f}%" if ev > 0 else "0.0%",
        "verdict": value_verdict(ev),
    }


def value_verdict(ev: float, threshold: float = 0.05) -> str:
    """Veredicto: VALUE BET si EV > umbral, NO VALUE si negativo."""
    if ev >= threshold:
        return "✅ VALUE BET"
    elif ev >= 0:
        return "🔶 NEUTRO"
    else:
        return "❌ NO VALUE"


def scan_value_bets(pred: dict, odds: dict) -> list[dict]:
    """
    Escanea todos los mercados en busca de EV+ y calcula el Stake según Criterio de Kelly Fraccionado.
    pred: salida de full_match_prediction + micromarkets
    odds: {"home": cuota, "draw": cuota, "away": cuota} (cuotas casa de apuestas)
    """
    results = []

    markets = [
        ("1 (Local Gana)", pred.get("home", 0), odds.get("home", 0)),
        ("X (Empate)",     pred.get("draw", 0), odds.get("draw", 0)),
        ("2 (Visitante)",  pred.get("away", 0), odds.get("away", 0)),
        ("Over 2.5",       pred.get("over_2_5", 0), odds.get("over", 0)),
        ("BTTS Sí",        pred.get("btts_yes", 0), odds.get("btts", 0)),
    ]

    for name, prob, bookie_odd in markets:
        if bookie_odd and bookie_odd > 1:
            ev = compute_ev(prob, bookie_odd)
            impl_prob = round(1 / bookie_odd, 4) if bookie_odd > 0 else 0
            kelly = compute_fractional_kelly(prob, bookie_odd, fraction=0.25)
            results.append({
                "Mercado": name,
                "Prob Modelo": f"{prob * 100:.1f}%",
                "Prob Implícita": f"{impl_prob * 100:.1f}%",
                "Cuota Justa": compute_fair_odds(prob),
                "Cuota Casa": bookie_odd,
                "EV%": f"{ev * 100:.1f}%",
                "Stake (Kelly)": f"{kelly * 100:.1f}%" if ev > 0 else "0.0%",
                "Veredicto": value_verdict(ev),
                "ev_raw": ev,
                "kelly_raw": kelly,
            })

    results.sort(key=lambda x: x["ev_raw"], reverse=True)
    return results


# =============================================================================
# UTILIDAD: Resumen del escáner de partidos
# =============================================================================

def scan_weekly_fixtures(div: str, odds_dict: dict = None) -> list[dict]:
    """
    Escanea los partidos de la semana para una liga y retorna ranking por EV+.
    odds_dict: { (home, away): {"home": X, "draw": X, "away": X} }
    """
    df = load_matches(div)
    fixtures = WEEKLY_FIXTURES.get(div, [])
    summary = []

    for home, away in fixtures:
        if home not in df["HomeTeam"].values and home not in df["AwayTeam"].values:
            continue
        pred = full_match_prediction(df, home, away)
        best_odds = {"home": 0, "draw": 0, "away": 0}
        if odds_dict and (home, away) in odds_dict:
            best_odds = odds_dict[(home, away)]

        # Probabilidad más alta → apuesta principal
        best_market = max(
            [("1", pred["home"]), ("X", pred["draw"]), ("2", pred["away"])],
            key=lambda x: x[1]
        )
        ev_main = compute_ev(best_market[1], best_odds.get(
            {"1": "home", "X": "draw", "2": "away"}[best_market[0]], 0
        ))

        summary.append({
            "Partido": f"{home} vs {away}",
            "xG": f"{pred['xg_home']:.2f} - {pred['xg_away']:.2f}",
            "Pronóstico": best_market[0],
            "Prob%": f"{best_market[1] * 100:.1f}%",
            "EV%": f"{ev_main * 100:.1f}%" if ev_main > -1 else "N/A",
            "ev_raw": ev_main if ev_main > -1 else -99,
        })

    summary.sort(key=lambda x: x["ev_raw"], reverse=True)
    return summary
