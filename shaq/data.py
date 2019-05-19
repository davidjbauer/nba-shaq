import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import time
from collections import defaultdict
from nba_api.stats.endpoints import teamyearbyyearstats, teamplayeronoffsummary
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import teams
import json

earliest_season = 1980
latest_season = 2019


def download_current_team_stats():
    """ Downloads up-to-date team data for the current season, cleans it,
        and saves it in a json file "data/team-stats.json"
    """
    team_df = pd.DataFrame(teams.get_teams())
    team_id_pairing = team_df[['abbreviation', 'id']]
    team_id_pairing.columns = ['Tm', 'TEAM_ID']

    stat_columns = [
        "GP",
        "WINS",
        "LOSSES",
        "WIN_PCT",
        "CONF_RANK",
        "DIV_RANK",
        "PO_WINS",
        "PO_LOSSES",
        "CONF_COUNT",
        "DIV_COUNT",
        "NBA_FINALS_APPEARANCE",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "PF",
        "STL",
        "TOV",
        "BLK",
        "PTS",
        "PTS_RANK",
    ]
    team_df = pd.DataFrame()
    for t in teams.get_teams():
        t_id = t["id"]
        # print(t_id, end = ', ')
        time.sleep(1)
        stat_json = json.loads(
            teamyearbyyearstats.TeamYearByYearStats(int(t_id)).get_json()
        )
        stat_rows = stat_json["resultSets"][0]["rowSet"]
        stat_headers = stat_json["resultSets"][0]["headers"]
        team_df = team_df.append(pd.DataFrame(stat_rows))
    team_df.columns = stat_headers

    team_stats_id_merged = pd.merge(team_df, team_id_pairing)
    team_stats = team_stats_id_merged.drop(
        columns=["TEAM_ID", "TEAM_CITY", "TEAM_NAME"]
    )
    team_stats = team_stats[["Tm", "YEAR"] + stat_columns]

    fixed_years = team_stats[["YEAR"]].apply(
        lambda x: int(x.to_string().split(" ")[-1].split("-")[0]) + 1, axis=1
    )
    team_stats[["YEAR"]] = fixed_years
    team_stats.to_csv("data/nba-api-team-stats.csv")
    team_stats = pd.read_csv("data/nba-api-team-stats.csv")
    team_stats = team_stats.drop(columns=["Unnamed: 0"])
    team_stats = team_stats[team_stats.YEAR >= earliest_season]
    team_stats.columns = ["Tm", "Year"] + stat_columns
    team_stats.to_json("data/team-stats.json")


def get_df_from_html(html_file):
    """ Given an HTML file from basketball-reference.com containing
        a table, parse the table into a pandas DataFrame.

        Input: html_file - file name of the HTML file
        Output: df - the DataFrame-ified table
    """
    with open(html_file) as html_doc:
        soup = BeautifulSoup(html_doc, "html.parser")
    table = soup.find("table")
    table_body = table.find("tbody")
    rows = table_body.find_all("tr")
    table_data = []
    for row in rows:
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        table_data.append([ele for ele in cols])

    df = pd.DataFrame(table_data)

    table_head = table.find("thead")
    headers = table_head.find_all("th")
    header_data = []
    cols = table_head.find_all("th")
    cols = [ele.text.strip() for ele in cols]
    header_data.append([ele for ele in cols])
    header_data = header_data[0]
    header_data.remove("Rk")
    df.columns = header_data

    return df


def process_soupy_df(df, yr):
    """
    """
    df["Year"] = yr
    df.to_json(f"data/scraped-player-data-{yr}.json")
    df = pd.read_json(f"data/scraped-player-data-{yr}.json")
    df.replace("", np.nan)
    df = df.loc[df.MP > 300]
    df = df.fillna(0)
    return df


def download_current_player_stats():
    """
    """
    stats_url = f"https://www.basketball-reference.com/leagues/NBA_{latest_season}_per_poss.html"
    adv_stats_url = f"https://www.basketball-reference.com/leagues/NBA_{latest_season}_advanced.html"

    stats_fname = f"data/{latest_season}_per_poss.html"
    adv_stats_fname = f"data/{latest_season}_adv.html"
    urllib.request.urlretrieve(stats_url, filename=stats_fname)
    urllib.request.urlretrieve(adv_stats_url, filename=adv_stats_fname)
    stats_df = get_df_from_html(stats_fname)
    stats_df = process_soupy_df(stats_df, latest_season)

    urllib.request.urlretrieve(adv_stats_url)
    adv_stats_df = get_df_from_html(adv_stats_fname)
    adv_stats_df["Year"] = latest_season
    adv_stats_df = adv_stats_df[
        [
            "Player",
            "Year",
            "Tm",
            "PER",
            "TS%",
            "3PAr",
            "FTr",
            "ORB%",
            "DRB%",
            "TRB%",
            "AST%",
            "STL%",
            "BLK%",
            "TOV%",
            "USG%",
            "OWS",
            "DWS",
            "WS",
            "WS/48",
            "OBPM",
            "DBPM",
            "BPM",
            "VORP",
        ]
    ]
    combined_stats = pd.merge(stats_df, adv_stats_df, on=["Player", "Year", "Tm"])
    combined_stats.to_json(f"data/{latest_season}-player.json")


def merge_player_team_stats():
    """
    """
    team_df = pd.read_json("data/team-stats.json")
    player_df = pd.read_json(f"data/{latest_season}-player.json")
    combined_df = pd.merge(player_df, team_df, on=["Tm", "Year"])
    combined_df.to_json(f"data/{latest_season}-combined.json")


def download_current_stats():
    download_current_team_stats()
    download_current_player_stats()
    merge_player_team_stats()
