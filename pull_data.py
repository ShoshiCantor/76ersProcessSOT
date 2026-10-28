"""
pull_data.py  --  clean data pull for the Sixers Ship of Theseus project

Everything here comes from the NBA's public stats site (via the nba_api library).
Nothing is typed by hand, so every run rebuilds all three files from scratch:

  data/player_seasons.csv  one row per player per season (jersey, GP, MPG, PPG...)
  data/team_seasons.csv    one row per season (wins, losses, win %)
  data/players.csv         one row per player (first/last season, games as a Sixer,
                           draft year/round/pick/team, headshot URL)

Setup:  pip install nba_api pandas
Run:    python pull_data.py
"""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    commonteamroster,
    drafthistory,
    leaguedashplayerstats,
    teamyearbyyearstats,
)

# ---------------- settings you can change ----------------
SIXERS_ID = 1610612755
FIRST_START_YEAR = 2013   # 2013 -> the 2013-14 season
LAST_START_YEAR = 2025    # 2025 -> the 2025-26 season. Raise it each new season.
OUT = Path("data")
# ---------------------------------------------------------


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


SEASONS = [season_label(y) for y in range(FIRST_START_YEAR, LAST_START_YEAR + 1)]


def with_retry(fn, tries=3):
    """stats.nba.com is flaky; retry a few times before giving up."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001
            print(f"  attempt {attempt} failed: {err}")
            time.sleep(3 * attempt)
    raise RuntimeError("NBA API kept failing. Try again later or from another network.")


def pull_player_seasons() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        print(f"Pulling {season} ...")
        stats = with_retry(lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            team_id_nullable=SIXERS_ID,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            timeout=60,
        ).get_data_frames()[0])
        time.sleep(1)
        roster = with_retry(lambda: commonteamroster.CommonTeamRoster(
            team_id=SIXERS_ID, season=season, timeout=60,
        ).get_data_frames()[0])
        time.sleep(1)

        stats = stats.rename(columns={
            "PLAYER_ID": "nba_id", "PLAYER_NAME": "name", "GP": "gp", "MIN": "mpg",
            "PTS": "ppg", "REB": "rpg", "AST": "apg", "STL": "spg", "BLK": "bpg",
            "FG_PCT": "fg_pct", "FG3_PCT": "fg3_pct", "FT_PCT": "ft_pct",
        })
        keep = ["nba_id", "name", "gp", "mpg", "ppg", "rpg", "apg",
                "spg", "bpg", "fg_pct", "fg3_pct", "ft_pct"]
        stats = stats[[c for c in keep if c in stats.columns]].copy()
        stats["source"] = "stats"

        # Players on the roster who never played (Embiid 2014-15, Simmons 2016-17...)
        # are missing from the stats table, so add them from the roster with gp = 0.
        missing = roster[~roster["PLAYER_ID"].isin(stats["nba_id"])]
        extra = pd.DataFrame({
            "nba_id": missing["PLAYER_ID"], "name": missing["PLAYER"],
            "gp": 0, "source": "roster_only",
        })
        season_df = pd.concat([stats, extra], ignore_index=True)

        # Jersey number comes from the roster list. A player who left mid-season
        # may not be on it, so his number is blank.
        season_df["jersey"] = season_df["nba_id"].map(
            dict(zip(roster["PLAYER_ID"], roster["NUM"].astype(str))))
        season_df.insert(2, "season", season)
        frames.append(season_df)

    out = pd.concat(frames, ignore_index=True)
    out["gp"] = out["gp"].fillna(0).astype(int)

    # Jersey numbers come from each season's END-of-season roster. Anyone traded or
    # waived mid-season is not on that list, so his jersey is blank for that season on
    # purpose: the page shows a number only for players who finished the year as a Sixer.
    out["jersey"] = out["jersey"].astype("string").str.strip().replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    return out


def pull_team_seasons() -> pd.DataFrame:
    df = with_retry(lambda: teamyearbyyearstats.TeamYearByYearStats(
        team_id=SIXERS_ID, per_mode_simple="Totals",
        season_type_all_star="Regular Season", timeout=60,
    ).get_data_frames()[0])
    df = df[df["YEAR"].isin(SEASONS)]
    df = df.rename(columns={"YEAR": "season", "WINS": "wins",
                            "LOSSES": "losses", "WIN_PCT": "win_pct"})
    return df[["season", "wins", "losses", "win_pct"]].reset_index(drop=True)


def pull_draft() -> pd.DataFrame:
    """Every NBA draft pick ever (one call). draft_team is the team that MADE the
    pick, so a draft-night trade shows up as the other team (e.g. Saric: ORL)."""
    df = with_retry(lambda: drafthistory.DraftHistory(
        league_id="00", timeout=60).get_data_frames()[0])
    df = df.rename(columns={"PERSON_ID": "nba_id", "SEASON": "draft_year",
                            "ROUND_NUMBER": "draft_round", "OVERALL_PICK": "draft_pick",
                            "TEAM_ABBREVIATION": "draft_team"})
    df["nba_id"] = df["nba_id"].astype(int)
    return (df[["nba_id", "draft_year", "draft_round", "draft_pick", "draft_team"]]
            .drop_duplicates("nba_id"))


def build_players(player_seasons: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    players = (player_seasons.groupby("nba_id")
               .agg(name=("name", "last"),
                    first_season=("season", "min"),
                    last_season=("season", "max"),
                    seasons_on_roster=("season", "nunique"),
                    games_as_sixer=("gp", "sum"))
               .reset_index())
    players = players.merge(draft, on="nba_id", how="left")
    players["draft_year"] = players["draft_year"].apply(
        lambda v: "Undrafted" if pd.isna(v) else str(v))
    for col in ["draft_round", "draft_pick"]:
        players[col] = players[col].apply(lambda v: "" if pd.isna(v) else str(int(v)))
    players["draft_team"] = players["draft_team"].fillna("")
    players["headshot_url"] = ("https://cdn.nba.com/headshots/nba/latest/1040x760/"
                               + players["nba_id"].astype(str) + ".png")
    return players.sort_values(["first_season", "name"]).reset_index(drop=True)


def main():
    OUT.mkdir(exist_ok=True)
    player_seasons = pull_player_seasons()
    teams = pull_team_seasons()
    players = build_players(player_seasons, pull_draft())

    player_seasons.to_csv(OUT / "player_seasons.csv", index=False, encoding="utf-8-sig")
    teams.to_csv(OUT / "team_seasons.csv", index=False, encoding="utf-8-sig")
    players.to_csv(OUT / "players.csv", index=False, encoding="utf-8-sig")

    print(f"\nDone. {len(players)} players, {len(player_seasons)} player-seasons, "
          f"{len(teams)} team seasons.")
    print(f"Player-seasons with a jersey number (finished the year on the roster): {(player_seasons['jersey'].notna()).sum()} "
          f"of {len(player_seasons)} player-seasons.")
    print(f"Files are in {OUT.resolve()}")


if __name__ == "__main__":
    main()
