"""
Módulo de Mercado en Vivo — The Odds API para Sports Control Center
Extrae cuotas de 'h2h' (1X2) y 'totals' (Over/Under 2.5) en tiempo real,
optimizando para cazar ineficiencias de cuotas (Best Odds).
Protegido con caché st.cache_data(ttl=300) para preservar la cuota de 500 peticiones.
"""

import requests
import re
import difflib
import os
import json
import datetime
import streamlit as st

try:
    from zoneinfo import ZoneInfo
    MADRID_TZ = ZoneInfo("Europe/Madrid")
except Exception:
    import pytz
    MADRID_TZ = pytz.timezone("Europe/Madrid")

CACHE_FILE = os.path.join(os.path.dirname(__file__), "cached_fixtures.json")

def get_odds_api_key() -> str:
    """
    Lectura segura de la clave de The Odds API con soporte para:
    1. st.secrets["ODDS_API_KEY"] (Streamlit Community Cloud)
    2. Variable de entorno ODDS_API_KEY
    3. Fallback a clave local segura
    """
    try:
        if hasattr(st, "secrets"):
            if "ODDS_API_KEY" in st.secrets:
                return str(st.secrets["ODDS_API_KEY"])
            elif hasattr(st.secrets, "get") and st.secrets.get("ODDS_API_KEY"):
                return str(st.secrets.get("ODDS_API_KEY"))
    except Exception:
        pass
    env_key = os.environ.get("ODDS_API_KEY")
    if env_key:
        return env_key
    return "2f60499434eeb4d63b03a12bd195938e"

API_KEY = get_odds_api_key()
BASE_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"

def parse_and_convert_to_spain(commence_time_str: str) -> tuple[datetime.datetime | None, str]:
    """
    Convierte una fecha ISO UTC de The Odds API a datetime con zona horaria Europe/Madrid
    y la formatea como '%d/%m %H:%M'.
    """
    if not commence_time_str:
        return None, ""
    try:
        dt = datetime.datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        spain_dt = dt.astimezone(MADRID_TZ)
        return spain_dt, spain_dt.strftime("%d/%m %H:%M")
    except Exception:
        return None, commence_time_str

SPORT_KEYS = {
    "SP1": "soccer_spain_la_liga",
    "SP2": "soccer_spain_segunda_division",
    "E0":  "soccer_epl",
    "I1":  "soccer_italy_serie_a",
    "D1":  "soccer_germany_bundesliga",
    "F1":  "soccer_france_ligue_one",
    "P1":  "soccer_portugal_primeira_liga",
}


# Casas recreativas / Soft Bookies requeridas
SOFT_BOOKMAKERS = [
    "bet365", "winamax", "betway", "bwin", "codere",
    "sportium", "retabet", "versus"
]
SOFT_BOOKMAKERS_PARAM = "bet365,winamax,betway,bwin,codere,sportium,retabet,versus"


def _normalize_name(s: str) -> str:
    """Normaliza nombres de equipos eliminando prefijos/sufijos habituales."""
    s = str(s).lower()
    for w in ["cf", "fc", "cd", "ca", "real", "de", "club", "vigo", "deportivo", "balompie"]:
        s = re.sub(r"\b" + w + r"\b", "", s)
    return re.sub(r"[^a-z0-9]", "", s).strip()


def _match_team(target: str, candidate: str) -> bool:
    """Compara si dos nombres de equipo se corresponden usando normalización y similitud."""
    t = _normalize_name(target)
    c = _normalize_name(candidate)
    if not t or not c:
        return False
    if t in c or c in t:
        return True
    return difflib.SequenceMatcher(None, t, c).ratio() >= 0.58


@st.cache_data(ttl=300, show_spinner=False)
def fetch_league_odds(div_code: str) -> tuple[list[dict], str]:
    """
    Descarga cuotas para una liga desde The Odds API.
    Caché de 5 minutos (300 segundos) para no agotar la cuota mensual.
    Retorna (partidos, peticiones_restantes).
    """
    sport_key = SPORT_KEYS.get(div_code)
    if not sport_key:
        return [], "Liga no soportada por The Odds API"

    url = BASE_URL.format(sport=sport_key)
    params = {
        "apiKey": get_odds_api_key(),
        "regions": "eu,uk",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        remaining = resp.headers.get("x-requests-remaining", "?")
        if resp.status_code == 200:
            data = resp.json()
            allowed_bookies = ['bet365', 'bwin', 'betway', 'williamhill', '888sport', 'unibet', 'betfair_ex_eu', 'winamax', 'marathonbet']
            # Filtro local estricto sobre el JSON recibido
            for m in data:
                m["bookmakers"] = [
                    b for b in m.get("bookmakers", [])
                    if any(ab in b.get("key", "").lower() or ab in b.get("title", "").lower() for ab in allowed_bookies)
                ]
            returned_bookies = sorted(list({b["key"] for m in data for b in m.get("bookmakers", [])}))
            print(f"📡 [The Odds API Audit] Casas tras filtro local estricto: {returned_bookies}")
            return data, remaining
        else:
            return [], remaining
    except Exception as e:
        return [], str(e)


def find_match_live_odds(div_code: str, home_team: str, away_team: str) -> dict | None:
    """
    Busca el enfrentamiento actual en los datos en vivo y extrae las mejores cuotas (Best Odds)
    y el consenso entre bookmakers.
    """
    matches, remaining = fetch_league_odds(div_code)
    if not matches:
        return None

    # Buscar el partido que coincida con local y visitante
    matched_event = None
    for m in matches:
        api_h = m.get("home_team", "")
        api_a = m.get("away_team", "")
        if _match_team(home_team, api_h) and _match_team(away_team, api_a):
            matched_event = m
            break
        # También chequear si vienen invertidos por error de localía
        elif _match_team(home_team, api_a) and _match_team(away_team, api_h):
            matched_event = m
            break

    if not matched_event:
        return None

    api_home = matched_event.get("home_team", "")
    api_away = matched_event.get("away_team", "")

    allowed_bookies = ['bet365', 'bwin', 'betway', 'williamhill', '888sport', 'unibet', 'betfair_ex_eu', 'winamax', 'marathonbet']
    home_prices = []
    draw_prices = []
    away_prices = []
    over_prices = []
    under_prices = []
    scanned_bookies = set()

    for bm in matched_event.get("bookmakers", []):
        bm_key = bm.get("key", "").lower()
        bm_title = bm.get("title", "").lower()
        if not any(ab in bm_key or ab in bm_title for ab in allowed_bookies):
            continue
        bm_name = bm.get("title", bm.get("key", "Soft Bookie"))
        scanned_bookies.add(bm_name)
        for market in bm.get("markets", []):
            m_key = market.get("key")
            if m_key == "h2h":
                for o in market.get("outcomes", []):
                    price = float(o.get("price", 0))
                    if price <= 1.0:
                        continue
                    o_name = o.get("name")
                    if o_name == api_home or _match_team(home_team, o_name):
                        home_prices.append((price, bm_name))
                    elif o_name == "Draw":
                        draw_prices.append((price, bm_name))
                    elif o_name == api_away or _match_team(away_team, o_name):
                        away_prices.append((price, bm_name))

            elif m_key == "totals":
                for o in market.get("outcomes", []):
                    price = float(o.get("price", 0))
                    point = float(o.get("point", 0))
                    if price <= 1.0 or point != 2.5:
                        continue
                    if o.get("name") == "Over":
                        over_prices.append((price, bm_name))
                    elif o.get("name") == "Under":
                        under_prices.append((price, bm_name))

    if not (home_prices or draw_prices or away_prices):
        return None

    best_h, bm_h = max(home_prices, key=lambda x: x[0]) if home_prices else (2.10, "N/A")
    best_x, bm_x = max(draw_prices, key=lambda x: x[0]) if draw_prices else (3.30, "N/A")
    best_a, bm_a = max(away_prices, key=lambda x: x[0]) if away_prices else (3.60, "N/A")
    best_o, bm_o = max(over_prices, key=lambda x: x[0]) if over_prices else (1.90, "N/A")
    best_u, bm_u = max(under_prices, key=lambda x: x[0]) if under_prices else (1.90, "N/A")

    raw_ct = matched_event.get("commence_time", "")
    _, formatted_ct = parse_and_convert_to_spain(raw_ct)

    return {
        "home": round(best_h, 2),
        "draw": round(best_x, 2),
        "away": round(best_a, 2),
        "over": round(best_o, 2),
        "under": round(best_u, 2),
        "btts": 1.85,  # Valor de referencia
        "bm_home": bm_h,
        "bm_draw": bm_x,
        "bm_away": bm_a,
        "bm_over": bm_o,
        "api_match": f"{api_home} vs {api_away}",
        "commence_time": formatted_ct if formatted_ct else raw_ct,
        "commence_time_raw": raw_ct,
        "num_bookmakers": len(scanned_bookies),
        "remaining_calls": remaining,
    }


def _is_cache_valid_for_week(cache_entry: dict) -> bool:
    """
    Comprueba si la entrada de caché corresponde a la misma semana actual
    (mismo año y misma semana del calendario ISO, con menos de 7 días de antigüedad).
    """
    cached_time_str = cache_entry.get("timestamp")
    if not cached_time_str:
        return False
    try:
        dt = datetime.datetime.fromisoformat(cached_time_str)
        now = datetime.datetime.now()
        same_iso_week = (dt.isocalendar()[:2] == now.isocalendar()[:2])
        is_fresh = (now - dt).total_seconds() < 7 * 86400
        return same_iso_week and is_fresh
    except Exception:
        return False


def get_upcoming_fixtures(sport_key: str = 'soccer_spain_la_liga', force_refresh: bool = False) -> list[tuple[str, str]]:
    """
    Obtiene los próximos partidos para la liga solicitada.
    Acepta tanto div_code ('SP1', 'E0', etc.) como sport_key ('soccer_spain_la_liga').
    Implementa persistencia semanal en cached_fixtures.json para no consumir la API
    más de una vez por semana.
    Devuelve lista de tuplas (equipo_local, equipo_visitante).
    """
    # Resolver div_code y clave de caché
    if sport_key in SPORT_KEYS:
        div_code = sport_key
    else:
        inv_map = {v: k for k, v in SPORT_KEYS.items()}
        div_code = inv_map.get(sport_key, "SP1")

    cache_key = div_code

    cache_data = {}
    if not force_refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            entry = cache_data.get(cache_key, {})
            if _is_cache_valid_for_week(entry):
                cached_items = entry.get("fixtures", [])
                if cached_items:
                    fixtures = []
                    for item in cached_items:
                        if len(item) >= 3:
                            fixtures.append((item[0], item[1], item[2]))
                        elif len(item) == 2:
                            fixtures.append((item[0], item[1], ""))
                    if fixtures:
                        print(f"📦 [Fixture Cache] Cargados {len(fixtures)} partidos de {cache_key} desde archivo local (0 llamadas API).")
                        return fixtures
        except Exception:
            cache_data = {}
    elif os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except Exception:
            cache_data = {}

    # Si se fuerza la recarga o no existe caché válida:
    if force_refresh:
        try:
            fetch_league_odds.clear()
        except Exception:
            pass

    print(f"🌐 [Fixture Fetch] Consultando partidos futuros para {cache_key}...")
    matches, _ = fetch_league_odds(div_code)
    fixtures = []

    if matches:
        now = datetime.datetime.now(datetime.timezone.utc)
        future_events = []
        for m in matches:
            ct_str = m.get("commence_time")
            if not ct_str:
                continue
            try:
                dt = datetime.datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
                if dt > now:
                    spain_dt, spain_fmt = parse_and_convert_to_spain(ct_str)
                    future_events.append((dt, spain_fmt, m))
            except Exception:
                continue

        future_events.sort(key=lambda x: x[0])

        if future_events:
            first_time = future_events[0][0]
            max_window_seconds = 3.5 * 86400  # Ventana máxima de 3.5 días
            seen = set()

            for dt, spain_fmt, m in future_events:
                if (dt - first_time).total_seconds() <= max_window_seconds:
                    h = m.get("home_team", "").strip()
                    a = m.get("away_team", "").strip()
                    if not h or not a or h == a:
                        continue
                    if h in seen or a in seen:
                        continue
                    seen.add(h)
                    seen.add(a)
                    fixtures.append((h, a, spain_fmt))

    if fixtures:
        cache_data[cache_key] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "fixtures": fixtures,
        }
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"💾 [Fixture Cache] Guardados {len(fixtures)} partidos de la jornada en {CACHE_FILE}.")
        except Exception as e:
            print(f"Error escribiendo en {CACHE_FILE}: {e}")

    return fixtures
