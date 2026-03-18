"""
Fetch League One match data with xG from FBref (StatsBomb xG data).

FBref provides free xG data for EFL League One via StatsBomb.
We scrape the scores & fixtures page for match-level xG.

Also supports loading standings + remaining fixtures as a fallback
when match-level data is unavailable.
"""

import time
import json
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# FBref League One 2025-26 scores & fixtures
FBREF_SCORES_URL = "https://fbref.com/en/comps/15/schedule/League-One-Scores-and-Fixtures"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LeagueOnePredictor/1.0; academic research)"
}


def fetch_fbref_matches(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch match results with xG from FBref's League One page.

    Returns DataFrame with columns:
        date, home_team, away_team, home_goals, away_goals, home_xg, away_xg
    """
    cache_path = DATA_DIR / "fbref_matches.csv"

    if use_cache and cache_path.exists():
        # Use cache if less than 6 hours old
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 6:
            print(f"Using cached data ({age_hours:.1f}h old)")
            return pd.read_csv(cache_path, parse_dates=["date"])

    print("Fetching match data from FBref...")
    resp = requests.get(FBREF_SCORES_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # FBref stores scores in a table with id "sched_..._1"
    table = soup.find("table", {"id": lambda x: x and x.startswith("sched_")})
    if table is None:
        raise RuntimeError("Could not find fixtures table on FBref page")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        # Skip spacer rows
        if tr.get("class") and "spacer" in tr.get("class", []):
            continue
        if tr.find("th", {"scope": "row"}) is None:
            continue

        cells = {}
        for td in tr.find_all(["td", "th"]):
            stat = td.get("data-stat", "")
            cells[stat] = td.get_text(strip=True)

        # We need: date, home team, away team, score, xG
        date = cells.get("date", "")
        home = cells.get("home_team", "") or cells.get("squad_a", "")
        away = cells.get("away_team", "") or cells.get("squad_b", "")
        score = cells.get("score", "")
        home_xg = cells.get("home_xg", "") or cells.get("xg_a", "")
        away_xg = cells.get("away_xg", "") or cells.get("xg_b", "")

        if not home or not away or not date:
            continue

        # Parse score (format: "2–1" or "2-1")
        home_goals, away_goals = None, None
        if score and ("–" in score or "-" in score):
            sep = "–" if "–" in score else "-"
            parts = score.split(sep)
            if len(parts) == 2:
                try:
                    home_goals = int(parts[0].strip())
                    away_goals = int(parts[1].strip())
                except ValueError:
                    pass

        # Parse xG
        try:
            home_xg_val = float(home_xg) if home_xg else None
        except ValueError:
            home_xg_val = None
        try:
            away_xg_val = float(away_xg) if away_xg else None
        except ValueError:
            away_xg_val = None

        rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_xg": home_xg_val,
            "away_xg": away_xg_val,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Save cache
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Fetched {len(df)} fixtures ({df['home_goals'].notna().sum()} played)")

    return df


def get_played_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Return only completed matches with valid xG data."""
    played = df.dropna(subset=["home_goals", "away_goals"]).copy()
    played["home_goals"] = played["home_goals"].astype(int)
    played["away_goals"] = played["away_goals"].astype(int)
    return played


def get_remaining_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Return fixtures not yet played."""
    return df[df["home_goals"].isna()].copy()


def load_manual_data(path: str = None) -> pd.DataFrame:
    """
    Load manually provided match data from a CSV file.

    This is the fallback if FBref scraping doesn't work.
    Expected columns: date, home_team, away_team, home_goals, away_goals, home_xg, away_xg

    For remaining fixtures, leave goals and xG columns empty.
    """
    if path is None:
        path = DATA_DIR / "matches.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_standings(path: str = None) -> pd.DataFrame:
    """
    Load league standings from CSV.

    Expected columns: team, played, won, drawn, lost, gf, ga, gd, points
    """
    if path is None:
        path = DATA_DIR / "standings.csv"
    return pd.read_csv(path)


def load_remaining_fixtures(path: str = None) -> pd.DataFrame:
    """
    Load remaining fixtures from CSV.

    Expected columns: date, home_team, away_team
    """
    if path is None:
        path = DATA_DIR / "remaining_fixtures.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def try_fetch_fbref(use_cache: bool = True):
    """Try FBref with cloudscraper as fallback for 403 errors."""
    cache_path = DATA_DIR / "fbref_matches.csv"

    if use_cache and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 6:
            print(f"Using cached FBref data ({age_hours:.1f}h old)")
            return pd.read_csv(cache_path, parse_dates=["date"])

    # Try cloudscraper first (handles Cloudflare), fall back to requests
    for attempt, fetcher in enumerate(_get_fetchers()):
        try:
            print(f"Fetching from FBref (attempt {attempt + 1})...")
            resp = fetcher(FBREF_SCORES_URL)
            if resp.status_code == 200:
                return _parse_fbref_html(resp.text, cache_path)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    raise RuntimeError(
        "Could not fetch from FBref. Use --standings mode instead:\n"
        "  python run_simulation.py --standings"
    )


def _get_fetchers():
    """Return list of HTTP fetcher functions to try."""
    fetchers = []

    # Try cloudscraper first
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper()
        fetchers.append(lambda url: scraper.get(url, timeout=30))
    except ImportError:
        pass

    # Fallback to requests
    fetchers.append(lambda url: requests.get(url, headers=HEADERS, timeout=30))

    return fetchers


def _parse_fbref_html(html: str, cache_path: Path) -> pd.DataFrame:
    """Parse FBref HTML and extract match data."""
    soup = BeautifulSoup(html, "lxml")

    # FBref sometimes hides tables in HTML comments
    import re
    comments = soup.find_all(string=lambda text: isinstance(text, str) and "sched_" in text)
    for comment in comments:
        comment_soup = BeautifulSoup(comment, "lxml")
        table = comment_soup.find("table", {"id": lambda x: x and x.startswith("sched_")})
        if table:
            break
    else:
        table = soup.find("table", {"id": lambda x: x and x.startswith("sched_")})

    if table is None:
        raise RuntimeError("Could not find fixtures table on FBref page")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if tr.get("class") and "spacer" in tr.get("class", []):
            continue
        if tr.find("th", {"scope": "row"}) is None:
            continue

        cells = {}
        for td in tr.find_all(["td", "th"]):
            stat = td.get("data-stat", "")
            cells[stat] = td.get_text(strip=True)

        date = cells.get("date", "")
        home = cells.get("home_team", "") or cells.get("squad_a", "")
        away = cells.get("away_team", "") or cells.get("squad_b", "")
        score = cells.get("score", "")
        home_xg = cells.get("home_xg", "") or cells.get("xg_a", "")
        away_xg = cells.get("away_xg", "") or cells.get("xg_b", "")

        if not home or not away or not date:
            continue

        home_goals, away_goals = None, None
        if score and ("–" in score or "-" in score):
            sep = "–" if "–" in score else "-"
            parts = score.split(sep)
            if len(parts) == 2:
                try:
                    home_goals = int(parts[0].strip())
                    away_goals = int(parts[1].strip())
                except ValueError:
                    pass

        try:
            home_xg_val = float(home_xg) if home_xg else None
        except ValueError:
            home_xg_val = None
        try:
            away_xg_val = float(away_xg) if away_xg else None
        except ValueError:
            away_xg_val = None

        rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_xg": home_xg_val,
            "away_xg": away_xg_val,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Fetched {len(df)} fixtures ({df['home_goals'].notna().sum()} played)")

    return df
