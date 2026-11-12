"""
Fase 4: Interfaz de control — Streamlit
Diseño minimalista estilo SaaS (Vercel/Apple), dark mode profesional y tooltips explicativos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import re
import importlib

sys.path.insert(0, os.path.dirname(__file__))

import engine
import live_market
import tactical_scout
try:
    importlib.reload(engine)
    importlib.reload(live_market)
    importlib.reload(tactical_scout)
except Exception:
    pass

from tactical_scout import load_team_tactics, load_squad_players, sync_tactical_database

from engine import (
    load_matches,
    load_understat_xg,
    get_teams,
    get_available_leagues,
    full_match_prediction,
    simulate_match_montecarlo,
    compute_micromarkets,
    scan_value_bets,
    scan_weekly_fixtures,
    compute_fair_odds,
    compute_ev,
    compute_fractional_kelly,
    calculate_ev_and_kelly,
    value_verdict,
    LEAGUE_DISPLAY,
    DISPLAY_TO_CODE,
    WEEKLY_FIXTURES,
    DB_PATH,
)

from live_market import (
    find_match_live_odds,
    get_upcoming_fixtures,
    _match_team,
    parse_and_convert_to_spain,
    MADRID_TZ,
)

# =============================================================================
# CONFIGURACIÓN Y ESTILOS SAAS MINIMALISTAS (ESTILO VERCEL / APPLE DARK)
# =============================================================================

st.set_page_config(
    page_title="Sports Control Center",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
  /* ── Tipografía del sistema (Apple San Francisco / Segoe UI) y fondo sólido ── */
  html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
  }
  .stApp {
    background-color: #111111;
    color: #f5f5f7;
  }
  section[data-testid="stSidebar"] {
    background-color: #161616;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
  }

  /* ── Sidebar typography & controls ── */
  .sidebar-title {
    color: #f5f5f7;
    font-size: 0.90rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding-bottom: 4px;
    margin-bottom: 8px;
  }

  /* ── Tabs minimalistas profesionales (tipo Apple / Vercel) ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 4px;
    margin-bottom: 20px;
  }
  .stTabs [data-baseweb="tab"] {
    height: 38px;
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    color: #86868b;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0 16px;
    transition: all 0.15s ease;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #f5f5f7;
    background-color: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.2);
  }
  .stTabs [aria-selected="true"] {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
    color: #f5f5f7 !important;
    font-weight: 600 !important;
  }

  /* ── Métricas estilo Apple Glassmorphism ── */
  [data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 14px 18px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
  }
  [data-testid="stMetricLabel"] {
    color: #86868b !important;
    font-size: 0.72rem !important;
    font-weight: 500;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }
  [data-testid="stMetricValue"] {
    color: #f5f5f7 !important;
    font-size: 1.45rem !important;
    font-weight: 600;
  }
  [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
  }

  /* ── Tablas limpias de alto contraste ── */
  .stDataFrame {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px;
    overflow: hidden;
  }

  /* ── Botones estilizados tipo SaaS ── */
  .stButton > button {
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background-color: rgba(255, 255, 255, 0.03);
    color: #f5f5f7;
    transition: all 0.15s ease;
  }
  .stButton > button:hover {
    border-color: rgba(255, 255, 255, 0.3);
    color: #ffffff;
    background-color: rgba(255, 255, 255, 0.06);
  }

  /* ── Chips de veredicto sutiles ── */
  .chip-value  { background:rgba(16, 185, 129, 0.12); color:#34d399; border:1px solid rgba(16, 185, 129, 0.25); border-radius:4px; padding:3px 8px; font-size:0.75rem; font-weight:600; }
  .chip-neutral{ background:rgba(245, 158, 11, 0.12); color:#fbbf24; border:1px solid rgba(245, 158, 11, 0.25); border-radius:4px; padding:3px 8px; font-size:0.75rem; font-weight:600; }
  .chip-no     { background:rgba(239, 68, 68, 0.12); color:#f87171; border:1px solid rgba(239, 68, 68, 0.25); border-radius:4px; padding:3px 8px; font-size:0.75rem; font-weight:600; }

  /* ── Cajas contenedoras (Glassmorphism con bordes y radio 12px) ── */
  .micro-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
  }
  .micro-title { color: #86868b; font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase; font-weight:600; }
  .micro-value { color: #f5f5f7; font-size: 1.35rem; font-weight: 600; margin: 4px 0; }
  .micro-sub   { color: #38bdf8; font-size: 0.75rem; }

  /* ── Subheaders limpios ── */
  h2, h3, .stSubheader {
    color: #f5f5f7 !important;
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
  }

  /* ── Barra de progreso sutil ── */
  .stProgress > div > div { background-color: #38bdf8 !important; }

  /* ── Ocultar elementos predeterminados de Streamlit ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
  hr { border-color: rgba(255, 255, 255, 0.08); }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)


# =============================================================================
# UTILIDADES UI
# =============================================================================

def render_verdict_badge(verdict: str) -> str:
    if "VALUE" in verdict or "VALOR" in verdict.upper():
        return f'<span class="chip-value">{verdict}</span>'
    elif "NEUTRO" in verdict.upper():
        return f'<span class="chip-neutral">{verdict}</span>'
    else:
        return f'<span class="chip-no">{verdict}</span>'


def micro_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""<div class="micro-card">
          <div class="micro-title">{title}</div>
          <div class="micro-value">{value}</div>
          <div class="micro-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def risk_level(pred: dict) -> tuple[str, str]:
    probs = [pred["home"], pred["draw"], pred["away"]]
    entropy = -sum(p * np.log(p + 1e-9) for p in probs)
    if entropy < 0.7:
        return "Bajo", "#34d399"
    elif entropy < 0.95:
        return "Moderado", "#fbbf24"
    else:
        return "Alto", "#f87171"


def check_cold_start_db() -> bool:
    """
    Verifica si la base de datos SQLite está construida y tiene datos tácticos y de partidos.
    Protección esencial contra el Cold Start de SQLite en Streamlit Community Cloud.
    """
    if not os.path.exists(DB_PATH):
        return False
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='team_tactics'")
        if cur.fetchone()[0] == 0:
            conn.close()
            return False
        cur.execute("SELECT count(*) FROM team_tactics")
        t_count = cur.fetchone()[0]
        cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='matches'")
        if cur.fetchone()[0] == 0:
            conn.close()
            return False
        cur.execute("SELECT count(*) FROM matches")
        m_count = cur.fetchone()[0]
        conn.close()
        return t_count > 0 and m_count > 0
    except Exception:
        return False


def check_db_exists() -> bool:
    return check_cold_start_db()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_understat_xg(div: str) -> pd.DataFrame:
    return load_understat_xg(div)


# =============================================================================
# SIDEBAR — PANEL DE CONTROL GLOBAL
# =============================================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">Sports Control Center</div>', unsafe_allow_html=True)
    st.caption("Plataforma de análisis cuantitativo deportivo")
    st.markdown("---")

    # Botones de Acción Globales (siempre disponibles incluso ante cold start)
    st.markdown('<div class="sidebar-title" style="font-size:0.72rem">Acciones globales</div>', unsafe_allow_html=True)
    if st.button("Actualizar Base Táctica", use_container_width=True, help="Sincroniza y reconstruye la base de datos táctica y plantillas en SQLite local."):
        with st.spinner("Sincronizando perfiles tácticos y plantillas en SQLite..."):
            sync_tactical_database()
            st.cache_data.clear()
            st.success("Base de datos táctica y plantillas actualizadas.")
            st.rerun()

    if st.button("Actualizar calendario de partidos", use_container_width=True, help="Consulta la API para sincronizar los próximos partidos de la jornada."):
        get_upcoming_fixtures("SP1", force_refresh=True)
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    db_ready = check_cold_start_db()
    if not db_ready:
        st.info("Inicialización requerida: pulsa 'Actualizar Base Táctica' para reconstruir el entorno.")
        div_code = "SP1"
        selected_league_name = "LaLiga Primera"
        teams = []
        home_team, away_team = "", ""
        injury_home, injury_away = False, False
        absent_players_home, absent_players_away = [], []
        df_league = pd.DataFrame()
    else:
        # 1. Selector de Liga
        st.markdown('<div class="sidebar-title" style="font-size:0.72rem">Competición</div>', unsafe_allow_html=True)
        available = get_available_leagues()

        if not available:
            st.error("No hay competiciones disponibles en la base de datos.")
            st.stop()

        selected_league_name = st.selectbox(
            "Competición",
            options=available,
            label_visibility="collapsed",
        )
        div_code = DISPLAY_TO_CODE.get(selected_league_name, "SP1")

        @st.cache_data(ttl=300, show_spinner=False)
        def _load(div):
            return load_matches(div)

        @st.cache_data(ttl=300, show_spinner=False)
        def _teams(div):
            return get_teams(div)

        df_league = _load(div_code)
        teams = _teams(div_code)

        st.markdown("---")

        # 2. Selector de Partido (Local y Visitante con horario peninsular)
        st.markdown('<div class="sidebar-title" style="font-size:0.72rem">Selección de partido</div>', unsafe_allow_html=True)
        upcoming_api = get_upcoming_fixtures(div_code)

        fixture_opts = []
        seen_side = set()

        if upcoming_api:
            for item in upcoming_api:
                api_h = item[0]
                api_a = item[1]
                match_time = item[2] if len(item) > 2 else ""
                matched_h = next((t for t in teams if _match_team(t, api_h)), api_h)
                matched_a = next((t for t in teams if _match_team(t, api_a)), api_a)
                if matched_h and matched_a and matched_h != matched_a and (matched_h, matched_a) not in seen_side:
                    seen_side.add((matched_h, matched_a))
                    time_label = f" ({match_time})" if match_time else ""
                    fixture_opts.append(f"{matched_h} vs {matched_a}{time_label}")

        if fixture_opts:
            preset = st.selectbox("Próximos partidos (jornada)", ["Seleccionar partido"] + fixture_opts,
                                  label_visibility="visible")
        else:
            preset = "Seleccionar partido"

        upcoming_teams = sorted(list({re.sub(r"\s*\(\d{2}/\d{2}\s+\d{2}:\d{2}\)$", "", p).split(" vs ")[0] for p in fixture_opts} | {re.sub(r"\s*\(\d{2}/\d{2}\s+\d{2}:\d{2}\)$", "", p).split(" vs ")[1] for p in fixture_opts}))
        team_opts = upcoming_teams if upcoming_teams else (teams if teams else ["Sin datos"])

        default_home = team_opts[0] if team_opts else "?"
        default_away = team_opts[1] if len(team_opts) > 1 else "?"
        if preset != "Seleccionar partido":
            clean_preset = re.sub(r"\s*\(\d{2}/\d{2}\s+\d{2}:\d{2}\)$", "", preset)
            parts = clean_preset.split(" vs ")
            if len(parts) == 2:
                dh, da = parts[0].strip(), parts[1].strip()
                if dh in team_opts:
                    default_home = dh
                if da in team_opts:
                    default_away = da
        elif fixture_opts:
            first_clean = re.sub(r"\s*\(\d{2}/\d{2}\s+\d{2}:\d{2}\)$", "", fixture_opts[0])
            first_pair = first_clean.split(" vs ")
            default_home, default_away = first_pair[0], first_pair[1]

        col_sel_h, col_sel_a = st.columns(2)
        with col_sel_h:
            home_team = st.selectbox("Local", team_opts,
                                     index=team_opts.index(default_home) if default_home in team_opts else 0)
        with col_sel_a:
            away_team = st.selectbox("Visitante", team_opts,
                                     index=team_opts.index(default_away) if default_away in team_opts else 0)

        if home_team == away_team:
            st.warning("Los equipos seleccionados deben ser distintos.")

        st.markdown("---")

        # 3. Selectores de Bajas Confirmadas (Local / Visitante)
        st.markdown('<div class="sidebar-title" style="font-size:0.72rem">Bajas y ausencias</div>', unsafe_allow_html=True)
        injury_home = st.checkbox(f"Baja de titular: {home_team[:12]}…" if len(home_team) > 12 else f"Baja de titular: {home_team}")
        injury_away = st.checkbox(f"Baja de titular: {away_team[:12]}…" if len(away_team) > 12 else f"Baja de titular: {away_team}")

        squad_h = load_squad_players(home_team)
        squad_a = load_squad_players(away_team)

        def _fmt_player_opt(p):
            return f"{p['player_name']} (-{p['importance_score']:.0f}% xG)"

        opts_h = [_fmt_player_opt(p) for p in squad_h]
        opts_a = [_fmt_player_opt(p) for p in squad_a]
        map_h = {_fmt_player_opt(p): p["player_name"] for p in squad_h}
        map_a = {_fmt_player_opt(p): p["player_name"] for p in squad_a}

        sel_h = st.multiselect(f"Bajas confirmadas ({home_team[:12]})", options=opts_h, help="Selecciona los jugadores descartados para ajustar su aportación ofensiva neta.")
        sel_a = st.multiselect(f"Bajas confirmadas ({away_team[:12]})", options=opts_a, help="Selecciona los jugadores descartados para ajustar su aportación ofensiva neta.")
        absent_players_home = [map_h[s] for s in sel_h]
        absent_players_away = [map_a[s] for s in sel_a]

        st.markdown("---")
        n_matches = len(df_league)
        st.caption(f"{n_matches} partidos registrados en base de datos · Liga: {div_code}")


# =============================================================================
# PROTECCIÓN DE COLD START (SQLite Epímero en Streamlit Community Cloud)
# =============================================================================
if not check_cold_start_db():
    st.warning("⚠️ El servidor se ha reiniciado o la base de datos está vacía. Ve a la barra lateral y pulsa 'Actualizar Base Táctica' para reconstruir el entorno.")
    st.stop()


# =============================================================================
# CÁLCULO PRINCIPAL Y ESTADO DEL ENFRENTAMIENTO
# =============================================================================

valid_match = home_team != away_team and home_team in teams and away_team in teams

if valid_match:
    try:
        pred = full_match_prediction(
            df_league, home_team, away_team,
            injury_home=injury_home, injury_away=injury_away,
            absent_players_home=absent_players_home,
            absent_players_away=absent_players_away
        )
        micro = compute_micromarkets(df_league, home_team, away_team)
        risk, risk_color = risk_level(pred)
    except Exception as e:
        st.error(f"Error en el cálculo predictivo: {e}")
        pred, micro, risk, risk_color = {}, {}, "Moderado", "#fbbf24"
else:
    pred, micro, risk, risk_color = {}, {}, "Moderado", "#fbbf24"


# Estado de cuotas en vivo por partido
live_key = f"live_{div_code}_{home_team}_{away_team}"
active_live = st.session_state.get(live_key)
if active_live:
    val_home = float(active_live.get("home", 2.10))
    val_draw = float(active_live.get("draw", 3.30))
    val_away = float(active_live.get("away", 3.60))
    val_over = float(active_live.get("over", 1.90))
    val_btts = float(active_live.get("btts", 1.85))
else:
    val_home, val_draw, val_away, val_over, val_btts = 2.10, 3.30, 3.60, 1.90, 1.85

current_odd_home = st.session_state.get(f"nh_{live_key}", val_home)
current_odd_draw = st.session_state.get(f"nx_{live_key}", val_draw)
current_odd_away = st.session_state.get(f"na_{live_key}", val_away)
current_odd_over = st.session_state.get(f"no_{live_key}", val_over)
current_odd_btts = st.session_state.get(f"nb_{live_key}", val_btts)

odds_input_base = {
    "home": current_odd_home,
    "draw": current_odd_draw,
    "away": current_odd_away,
    "over": current_odd_over,
    "btts": current_odd_btts,
}
value_rows = scan_value_bets(pred, odds_input_base) if pred else []
best_bet = max(value_rows, key=lambda x: x.get("ev_raw", -999)) if value_rows else None

# Horario peninsular del enfrentamiento seleccionado
current_match_time = ""
if 'upcoming_api' in locals() and upcoming_api:
    for item in upcoming_api:
        h, a = item[0], item[1]
        t = item[2] if len(item) > 2 else ""
        if (_match_team(home_team, h) and _match_team(away_team, a)) or (_match_team(home_team, a) and _match_team(away_team, h)):
            current_match_time = t
            break
if not current_match_time and active_live and active_live.get("commence_time"):
    current_match_time = active_live.get("commence_time")


# =============================================================================
# ENCABEZADO PRINCIPAL (MINIMALISTA)
# =============================================================================

st.markdown(
    f"""<div style='display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:12px;'>
      <div>
        <h1 style='color:#f8fafc;font-size:1.45rem;font-weight:700;margin:0;letter-spacing:-0.01em;'>
          Sports Control Center
        </h1>
        <p style='color:#94a3b8;font-size:0.82rem;margin:3px 0 0 0;'>
          {selected_league_name} &nbsp;·&nbsp;
          <b style='color:#ffffff;'>{home_team}</b> <span style='color:#64748b;'>(Elo {pred.get('home_elo', '—') if valid_match else '—'})</span>
          <span style='color:#38bdf8;margin:0 4px;'>vs</span>
          <b style='color:#ffffff;'>{away_team}</b> <span style='color:#64748b;'>(Elo {pred.get('away_elo', '—') if valid_match else '—'})</span>
          {f"&nbsp;·&nbsp; <span style='color:#38bdf8;'>Saque inicial: <b>{current_match_time}</b> (España)</span>" if current_match_time else ""}
        </p>
      </div>
      <div style='text-align:right;'>
        <span style='background:rgba(56,189,248,0.08);color:#38bdf8;border:1px solid rgba(56,189,248,0.25);border-radius:4px;padding:4px 10px;font-size:0.75rem;font-weight:600;'>
          {selected_league_name}
        </span>
      </div>
    </div>""",
    unsafe_allow_html=True,
)
st.markdown('<hr style="margin:4px 0 16px 0;"/>', unsafe_allow_html=True)

if not valid_match:
    st.warning("Selecciona dos equipos distintos de la misma liga en la barra lateral para continuar.")
    st.stop()


# =============================================================================
# ESTRUCTURA PRINCIPAL POR PESTAÑAS (TABS)
# =============================================================================

tab_radar, tab_match, tab_trading, tab_props = st.tabs([
    "Radar de jornada",
    "Pizarra y táctica",
    "Escáner de valor",
    "Micro-mercados",
])


# -----------------------------------------------------------------------------
# TAB 1: RADAR DE LA JORNADA
# -----------------------------------------------------------------------------
with tab_radar:
    st.subheader("Radar de favoritos · Ranking de la jornada", help="Ordenado de mayor a menor probabilidad matemática de victoria.")

    @st.cache_data(ttl=300, show_spinner=False)
    def _generate_league_radar(div):
        df_matches = load_matches(div)
        if df_matches.empty:
            return []

        upcoming = get_upcoming_fixtures(div)
        if not upcoming:
            return []

        league_teams = set(df_matches["HomeTeam"].dropna()) | set(df_matches["AwayTeam"].dropna())
        radar_data = []
        seen_teams = set()

        for item in upcoming:
            api_h = item[0]
            api_a = item[1]
            match_time = item[2] if len(item) > 2 else "—"
            matched_h = next((t for t in league_teams if _match_team(t, api_h)), api_h)
            matched_a = next((t for t in league_teams if _match_team(t, api_a)), api_a)
            if not matched_h or not matched_a or matched_h == matched_a:
                continue
            if matched_h in seen_teams or matched_a in seen_teams:
                continue
            seen_teams.add(matched_h)
            seen_teams.add(matched_a)

            pred_match = full_match_prediction(df_matches, matched_h, matched_a)
            options = [
                ("1", pred_match["home"]),
                ("X", pred_match["draw"]),
                ("2", pred_match["away"]),
            ]
            best_code, best_prob = max(options, key=lambda x: x[1])
            fair_odd = compute_fair_odds(best_prob)

            radar_data.append({
                "Horario": match_time if match_time else "—",
                "Partido": f"{matched_h} vs {matched_a}",
                "Goles esperados": f"{pred_match['xg_home']:.2f} - {pred_match['xg_away']:.2f}",
                "Pronóstico": best_code,
                "Probabilidad": f"{best_prob * 100:.1f}%",
                "Cuota justa": f"{fair_odd:.2f}",
                "_prob_raw": best_prob,
            })

        radar_data.sort(key=lambda x: x["_prob_raw"], reverse=True)
        return radar_data

    radar_matches = _generate_league_radar(div_code)

    if radar_matches:
        df_radar = pd.DataFrame(radar_matches)
        display_df = df_radar[["Horario", "Partido", "Goles esperados", "Pronóstico", "Probabilidad", "Cuota justa"]].copy()

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption("Horarios en hora peninsular española (Europe/Madrid). La cuota justa representa el valor matemático mínimo necesario antes de aplicar el margen de la casa de apuestas.")
    else:
        st.info(f"No hay suficientes datos de partidos para {selected_league_name}.")


# -----------------------------------------------------------------------------
# TAB 2: PIZARRA & TÁCTICA
# -----------------------------------------------------------------------------
with tab_match:
    col_t_title, col_t_time = st.columns([3, 1])
    with col_t_title:
        st.subheader("Resumen del enfrentamiento")
    with col_t_time:
        if current_match_time:
            st.markdown(
                f"<div style='text-align:right;color:#94a3b8;font-size:0.80rem;padding-top:4px;'>"
                f"Horario: <b style='color:#f5f5f7;'>{current_match_time}</b> (España)"
                f"</div>",
                unsafe_allow_html=True,
            )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        xg_label = f"{pred.get('xg_home', 1.0):.2f} - {pred.get('xg_away', 1.0):.2f}"
        elo_diff = pred.get("elo_diff", None)
        if elo_diff is not None:
            elo_sign = "+" if elo_diff >= 0 else ""
            xg_delta = f"Elo {elo_sign}{elo_diff:.0f} · {'Local superior' if elo_diff > 0 else ('Visitante superior' if elo_diff < 0 else 'Equilibrado')}"
        else:
            xg_delta = "Local superior" if pred.get("xg_home", 0) > pred.get("xg_away", 0) else ("Visitante superior" if pred.get("xg_away", 0) > pred.get("xg_home", 0) else "Equilibrado")
        st.metric(
            "xG Proyectado",
            xg_label,
            delta=xg_delta,
            help="Goles que el equipo debería marcar hoy basado en la calidad histórica de sus ocasiones.",
        )

    with c2:
        best_1x2 = max([("1", pred.get("home", 0.33)), ("X", pred.get("draw", 0.33)), ("2", pred.get("away", 0.33))], key=lambda x: x[1])
        st.metric(
            "Pronóstico",
            f"{best_1x2[0]} · {best_1x2[1]*100:.1f}%",
            delta=f"Empate {pred.get('draw', 0.33)*100:.1f}%",
            help="Resultado con mayor probabilidad matemática según el modelo de goles esperados.",
        )

    with c3:
        if best_bet:
            bet_kelly = str(best_bet.get("Stake (Kelly)", "0.0%"))
            st.metric(
                "Apuesta EV+",
                best_bet.get("Mercado", "Ninguna"),
                delta=f"{best_bet.get('EV%', '0%')} · Kelly: {bet_kelly}",
                help="Valor Esperado. Porcentaje de beneficio a largo plazo si repites esta apuesta.",
            )
        else:
            st.metric(
                "Apuesta EV+",
                "Sin cuotas",
                delta="Introduce cuotas",
                help="Valor Esperado. Porcentaje de beneficio a largo plazo si repites esta apuesta.",
            )

    with c4:
        st.metric(
            "Riesgo",
            risk,
            delta=f"Más de 2.5: {pred.get('over_2_5', 0.5)*100:.1f}%",
            help="Nivel de dispersión e incertidumbre según la entropía de probabilidades 1X2.",
        )

    # True xG de Understat
    _src = pred.get("xg_source", "goals_only")
    _txg_h = pred.get("true_xg_home")
    _txga_h = pred.get("true_xga_home")
    _txg_a = pred.get("true_xg_away")
    _txga_a = pred.get("true_xga_away")

    if _txg_h is not None and _txg_a is not None:
        st.markdown(
            f"""<div style='background:rgba(255, 255, 255, 0.03);border:1px solid rgba(255, 255, 255, 0.1);border-radius:12px;
                padding:10px 16px;margin:12px 0;font-family:monospace;font-size:0.78rem;'>
            <span style='color:#38bdf8;font-weight:600'>True xG · Métricas avanzadas Understat</span>
            &nbsp;|&nbsp;
            <b>{home_team[:18]}</b>:
                xG <span style='color:#38bdf8'>{_txg_h:.3f}</span>
                &nbsp;xGA <span style='color:#fbbf24'>{_txga_h:.3f}</span>
            &nbsp;&nbsp;
            <b>{away_team[:18]}</b>:
                xG <span style='color:#38bdf8'>{_txg_a:.3f}</span>
                &nbsp;xGA <span style='color:#fbbf24'>{_txga_a:.3f}</span>
            &nbsp;|&nbsp;
            <span style='color:#777'>Ponderación: 60% True xG + 40% Goles Reales</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Barra compacta de Duelo Táctico
    tac_info = pred.get("tactical_matchup", {})
    if tac_info:
        raw_ph = tac_info.get("possession_home", 50.0)
        raw_pa = tac_info.get("possession_away", 50.0)
        rel_ph = int(round((raw_ph / max(1.0, raw_ph + raw_pa)) * 100))
        rel_pa = 100 - rel_ph
        style_h = tac_info.get("style_home", "Vertical / Transición")
        style_a = tac_info.get("style_away", "Vertical / Transición")
        ppda_h = tac_info.get("ppda_home", 11.5)
        ppda_a = tac_info.get("ppda_away", 11.5)

        elo_h = pred.get("home_elo", 1500.0)
        elo_a = pred.get("away_elo", 1500.0)
        depth_h = int(round(35 + 50 * float(np.clip((elo_h - 1350.0) / 500.0, 0.0, 1.0))))
        depth_a = int(round(35 + 50 * float(np.clip((elo_a - 1350.0) / 500.0, 0.0, 1.0))))

        syn_note = tac_info.get("synergy_note", "")

        abs_notes_h = pred.get("home_abs_notes", [])
        abs_notes_a = pred.get("away_abs_notes", [])
        all_abs_notes = []
        if abs_notes_h:
            all_abs_notes.append(f"<b>{home_team[:12]}:</b> " + ", ".join(abs_notes_h))
        if abs_notes_a:
            all_abs_notes.append(f"<b>{away_team[:12]}:</b> " + ", ".join(abs_notes_a))

        synergy_html = f"<div style='margin-top:8px;padding-top:6px;border-top:1px dashed rgba(255,255,255,0.08);color:#38bdf8;font-size:0.74rem;'><b>Sinergia táctica:</b> {syn_note}</div>" if syn_note else ""
        bajas_html = f"<div style='margin-top:6px;padding-top:4px;border-top:1px dashed rgba(255,255,255,0.08);color:#fbbf24;font-size:0.74rem;'><b>Ajuste por bajas:</b> {' &nbsp;|&nbsp; '.join(all_abs_notes)}</div>" if all_abs_notes else ""

        duelo_html = f"""<div style='background:rgba(255, 255, 255, 0.03);border:1px solid rgba(255, 255, 255, 0.1);border-left:3px solid #38bdf8;border-radius:12px;padding:14px 16px;margin:12px 0;font-family:monospace;font-size:0.78rem;'>
<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
<span style='color:#f8fafc;font-weight:600;letter-spacing:0.02em;text-transform:uppercase;'>Duelo táctico y profundidad de plantilla</span>
<span style='color:#94a3b8;font-size:0.75rem;'>Posesión proyectada: <b style='color:#38bdf8;'>{rel_ph}%</b> vs <b style='color:#e2e8f0;'>{rel_pa}%</b></span>
</div>
<div style='display:grid;grid-template-columns: repeat(4, 1fr);gap:10px;'>
<div style='background:rgba(255,255,255,0.02);padding:8px 10px;border-radius:4px;'>
<div style='color:#64748b;font-size:0.65rem;text-transform:uppercase;'>Estilo local</div>
<div style='color:#ffffff;font-weight:600;font-size:0.82rem;'>{style_h}</div>
</div>
<div style='background:rgba(255,255,255,0.02);padding:8px 10px;border-radius:4px;'>
<div style='color:#64748b;font-size:0.65rem;text-transform:uppercase;'>Estilo visitante</div>
<div style='color:#ffffff;font-weight:600;font-size:0.82rem;'>{style_a}</div>
</div>
<div style='background:rgba(255,255,255,0.02);padding:8px 10px;border-radius:4px;'>
<div style='color:#64748b;font-size:0.65rem;text-transform:uppercase;'>Presión defensiva (PPDA)</div>
<div style='color:#fbbf24;font-weight:600;font-size:0.82rem;'>{ppda_h:.1f} <span style='color:#64748b;'>vs</span> {ppda_a:.1f}</div>
</div>
<div style='background:rgba(255,255,255,0.02);padding:8px 10px;border-radius:4px;'>
<div style='color:#64748b;font-size:0.65rem;text-transform:uppercase;'>Profundidad de plantilla</div>
<div style='color:#38bdf8;font-weight:600;font-size:0.82rem;'>{depth_h}% <span style='color:#64748b;'>vs</span> {depth_a}%</div>
</div>
</div>
{synergy_html}
{bajas_html}
<div style='color:#94a3b8;font-size:0.71rem;margin-top:6px;'>Posesión y PPDA: Mide la intensidad defensiva y el control del balón.</div>
</div>"""
        st.markdown(duelo_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Predicción de resultados", help="Probabilidades exactas calculadas combinando el modelo Poisson, corrección de marcadores bajos Dixon-Coles y simulación Montecarlo.")

    col_1x2, col_over = st.columns(2)
    with col_1x2:
        st.markdown("**Probabilidad de victoria (1X2)**")
        st.markdown(f"""
        | Resultado | Probabilidad | Cuota justa |
        |:---|:---:|:---:|
        | Victoria local · {home_team[:14]} | **{pred['home']*100:.1f}%** | {compute_fair_odds(pred['home'])} |
        | Empate | **{pred['draw']*100:.1f}%** | {compute_fair_odds(pred['draw'])} |
        | Victoria visitante · {away_team[:14]} | **{pred['away']*100:.1f}%** | {compute_fair_odds(pred['away'])} |
        """)
        st.progress(pred["home"], text=f"Local {pred['home']*100:.0f}%")
        st.progress(pred["draw"], text=f"Empate {pred['draw']*100:.0f}%")
        st.progress(pred["away"], text=f"Visitante {pred['away']*100:.0f}%")

    with col_over:
        st.markdown("**Probabilidad de goles (Over / BTTS)**")
        st.markdown(f"""
        | Mercado | Probabilidad | Proyección |
        |:---|:---:|:---:|
        | Más de 2.5 goles | **{pred['over_2_5']*100:.1f}%** | {'Favorable' if pred['over_2_5'] > 0.5 else 'Desfavorable'} |
        | Menos de 2.5 goles | **{pred['under_2_5']*100:.1f}%** | {'Favorable' if pred['under_2_5'] > 0.5 else 'Desfavorable'} |
        | Ambos marcan (Sí) | **{pred['btts_yes']*100:.1f}%** | {'Probable' if pred['btts_yes'] > 0.5 else 'Improbable'} |
        | Ambos marcan (No) | **{pred['btts_no']*100:.1f}%** | {'Probable' if pred['btts_no'] > 0.5 else 'Improbable'} |
        """)
        st.progress(pred["over_2_5"], text=f"Más de 2.5 goles: {pred['over_2_5']*100:.0f}%")
        st.progress(pred["btts_yes"], text=f"Ambos marcan: {pred['btts_yes']*100:.0f}%")


# -----------------------------------------------------------------------------
# TAB 3: ESCÁNER DE VALOR (TRADING Y VALUE BETTING)
# -----------------------------------------------------------------------------
with tab_trading:
    st.subheader("Cuotas de mercado en vivo", help="Compara las cuotas reales de las casas de apuestas con la cuota justa matemática del modelo.")

    col_btn_scan, col_stat_scan = st.columns([1, 2])
    with col_btn_scan:
        btn_scan = st.button("Escanear mercado en vivo", use_container_width=True, help="Consulta las cuotas actualizadas en The Odds API.")

    if btn_scan:
        with st.spinner("Consultando cuotas en vivo en The Odds API..."):
            try:
                odds_found = find_match_live_odds(div_code, home_team, away_team)
                if odds_found:
                    st.session_state[live_key] = odds_found
                    st.success(f"Cuotas en vivo detectadas ({odds_found['num_bookmakers']} casas analizadas)")
                else:
                    st.warning("No se encontraron cuotas en vivo para este enfrentamiento en The Odds API.")
            except Exception as e:
                st.error(f"Error en API: {e}")

    active_live = st.session_state.get(live_key)
    if active_live:
        ct_disp = active_live.get('commence_time', '')
        time_tag = f" &nbsp;·&nbsp; Horario: <b>{ct_disp}</b> (España)" if ct_disp else ""
        st.markdown(
            f"""<div style='background:rgba(56, 189, 248, 0.06);border:1px solid rgba(56, 189, 248, 0.2);border-radius:6px;
                padding:8px 14px;margin-bottom:12px;font-size:0.78rem;font-family:monospace;'>
            <span style='color:#38bdf8;font-weight:600'>En vivo:</span> {active_live.get('api_match')}{time_tag} &nbsp;|&nbsp;
            Casas analizadas: <b>{active_live.get('num_bookmakers')}</b> &nbsp;·&nbsp;
            Consultas API restantes: <b>{active_live.get('remaining_calls')}</b>
            </div>""",
            unsafe_allow_html=True,
        )
        val_home = float(active_live.get("home", 2.10))
        val_draw = float(active_live.get("draw", 3.30))
        val_away = float(active_live.get("away", 3.60))
        val_over = float(active_live.get("over", 1.90))
        val_btts = float(active_live.get("btts", 1.85))
    else:
        val_home, val_draw, val_away, val_over, val_btts = 2.10, 3.30, 3.60, 1.90, 1.85

    col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
    with col_q1:
        odd_home = st.number_input("Local (1)", min_value=1.01, max_value=50.0, value=val_home, step=0.05, key=f"nh_{live_key}", help="Cuota ofrecida por la casa de apuestas para la victoria local.")
    with col_q2:
        odd_draw = st.number_input("Empate (X)", min_value=1.01, max_value=50.0, value=val_draw, step=0.05, key=f"nx_{live_key}", help="Cuota ofrecida por la casa de apuestas para el empate.")
    with col_q3:
        odd_away = st.number_input("Visitante (2)", min_value=1.01, max_value=50.0, value=val_away, step=0.05, key=f"na_{live_key}", help="Cuota ofrecida por la casa de apuestas para la victoria visitante.")
    with col_q4:
        odd_over = st.number_input("Más de 2.5", min_value=1.01, max_value=20.0, value=val_over, step=0.05, key=f"no_{live_key}", help="Cuota ofrecida para más de 2.5 goles totales.")
    with col_q5:
        odd_btts = st.number_input("Ambos marcan", min_value=1.01, max_value=20.0, value=val_btts, step=0.05, key=f"nb_{live_key}", help="Cuota para que ambos equipos anoten al menos un gol.")

    odds_active = {
        "home": odd_home,
        "draw": odd_draw,
        "away": odd_away,
        "over": odd_over,
        "btts": odd_btts,
    }
    value_rows = scan_value_bets(pred, odds_active) if pred else []

    st.markdown("---")
    st.subheader("Escáner de apuestas con valor (EV+)", help="Valor Esperado. Porcentaje de beneficio a largo plazo si repites esta apuesta.")

    try:
        if value_rows:
            def _safe_format_kelly(val):
                if val is None or val == "":
                    return "0.0%"
                if isinstance(val, (int, float)):
                    return f"{val * 100:.1f}%"
                val_str = str(val).strip()
                if not val_str.endswith("%"):
                    try:
                        return f"{float(val_str) * 100:.1f}%"
                    except Exception:
                        return "0.0%"
                return val_str

            df_display = pd.DataFrame([
                {
                    "Mercado": str(r.get("Mercado", "-")),
                    "Prob. Modelo": str(r.get("Prob Modelo", "0.0%")),
                    "Prob. Casa": str(r.get("Prob Implícita", "0.0%")),
                    "Cuota Justa": r.get("Cuota Justa", "-"),
                    "Cuota Casa": r.get("Cuota Casa", "-"),
                    "Valor (EV%)": str(r.get("EV%", "0.0%")),
                    "Stake (Kelly)": _safe_format_kelly(r.get("Stake (Kelly)", r.get("kelly_pct", "0.0%"))),
                    "Veredicto": "Valor detectado" if "VALUE" in str(r.get("Veredicto", "")) else ("Neutro" if "NEUTRO" in str(r.get("Veredicto", "")) else "Sin valor"),
                }
                for r in value_rows
            ])

            st.dataframe(df_display, use_container_width=True, hide_index=True)
            st.caption("Cuota Justa: La cuota real matemática. Si la casa de apuestas paga más que esto, hay valor (EV+). · Stake (Kelly): Porcentaje de tu dinero que deberías apostar según el riesgo matemático.")
        else:
            st.info("Introduce cuotas de casas de apuestas arriba para activar el escáner.")
    except Exception as e:
        st.error(f"Error visualizando escáner de valor: {e}")

    st.markdown("---")
    st.subheader("Marcadores exactos más probables", help="La cuota real matemática. Si la casa de apuestas paga más que esto, hay valor (EV+).")

    top_scores = pred.get("top_scores", [])
    if top_scores:
        sc1, sc2, sc3 = st.columns(3)
        for idx, (score, pct) in enumerate(top_scores[:3]):
            col_target = [sc1, sc2, sc3][idx]
            with col_target:
                col_target.metric(
                    f"Marcador #{idx+1}",
                    score,
                    delta=f"{pct:.1f}% probabilidad",
                    help="La cuota real matemática. Si la casa de apuestas paga más que esto, hay valor (EV+).",
                )


# -----------------------------------------------------------------------------
# TAB 4: MICRO-MERCADOS
# -----------------------------------------------------------------------------
with tab_props:
    st.subheader("Micro-mercados proyectados", help="Proyecciones cuantitativas para eventos específicos del partido (córners, tarjetas y tiros).")

    mc_col1, mc_col2, mc_col3 = st.columns(3)

    with mc_col1:
        st.markdown("**Córners totales**")
        if micro.get("corners_expected_total") is not None:
            micro_card(
                "Total esperado",
                f"{micro['corners_expected_total']:.1f}",
                f"{home_team[:10]}: {micro['corners_expected_home']:.1f}  |  {away_team[:10]}: {micro['corners_expected_away']:.1f}"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "Más de 8.5",
                    f"{micro['corners_over_8_5']*100:.1f}%",
                    delta="Favorable" if micro["corners_over_8_5"] > 0.5 else "Baja prob",
                    help="Probabilidad calculada mediante Poisson para superar la línea de 8.5 córners.",
                )
            with col_b:
                st.metric(
                    "Más de 9.5",
                    f"{micro['corners_over_9_5']*100:.1f}%",
                    delta="Favorable" if micro["corners_over_9_5"] > 0.5 else "Baja prob",
                    help="Probabilidad calculada mediante Poisson para superar la línea de 9.5 córners.",
                )
        else:
            st.caption("Sin datos de córners en base de datos.")

    with mc_col2:
        st.markdown("**Tarjetas disciplinarias**")
        if micro.get("cards_expected_total") is not None:
            micro_card(
                "Total esperado",
                f"{micro['cards_expected_total']:.1f}",
                f"{home_team[:10]}: {micro['cards_expected_home']:.1f}  |  {away_team[:10]}: {micro['cards_expected_away']:.1f}"
            )
            st.metric(
                "Más de 4.5 tarjetas",
                f"{micro['cards_over_4_5']*100:.1f}%",
                delta="Favorable" if micro["cards_over_4_5"] > 0.5 else "Baja prob",
                help="Probabilidad calculada considerando intensidad de presión PPDA y estilos tácticos.",
            )
        else:
            st.caption("Sin datos de tarjetas en base de datos.")

    with mc_col3:
        st.markdown("**Tiros a puerta**")
        if micro.get("shots_expected_home") is not None:
            micro_card(
                f"{home_team[:14]} (Local)",
                f"{micro['shots_expected_home']:.1f}",
                "tiros a puerta esperados"
            )
            micro_card(
                f"{away_team[:14]} (Visitante)",
                f"{micro['shots_expected_away']:.1f}",
                "tiros a puerta esperados"
            )
        else:
            st.caption("Sin datos de tiros en base de datos.")


# =============================================================================
# PIE DE PÁGINA (MINIMALISTA)
# =============================================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """<div style='border-top:1px solid rgba(255, 255, 255, 0.06);padding-top:12px;color:#64748b;font-size:0.72rem;text-align:center;'>
    Sports Control Center · Interfaz profesional · Análisis estadístico y cuantitativo
    </div>""",
    unsafe_allow_html=True,
)
