"""
tools.py - Tool definitions (OpenAI function-calling schema) and execution dispatcher.
Wraps the existing CRAG mock_api KG interfaces.
"""
import json
import sys
import os
from typing import Any, Dict, List

# Add parent repo to path so we can import the CRAG client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.cragapi_wrapper import CRAG


# =============================================================================
# ANSWER TOOL (universal — signals end of ReAct loop)
# =============================================================================

ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Submit your final answer to the question. Call this tool when you have gathered enough information to answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your final answer. Be concise and precise."
                }
            },
            "required": ["answer"]
        }
    }
}


# =============================================================================
# OPEN DOMAIN TOOLS
# =============================================================================

OPEN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_search_entity_by_name",
            "description": "Search for entities by name in the encyclopedia knowledge graph. Returns at most 10 matching entity names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for entity name"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_get_entity",
            "description": "Get detailed information about a specific entity from the encyclopedia knowledge graph. Returns summary text, structured summary, and raw wiki content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "The exact entity name to look up"}
                },
                "required": ["entity"]
            }
        }
    },
]


# =============================================================================
# MOVIE DOMAIN TOOLS
# =============================================================================

MOVIE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "movie_get_person_info",
            "description": "Search for a person (actor/director) in the movie database via BM25. Returns info including name, acted/directed movies, birthday, oscar awards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_name": {"type": "string", "description": "Person name to search"}
                },
                "required": ["person_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "movie_get_movie_info",
            "description": "Search for a movie in the database via BM25. Returns info including title, release_date, budget, revenue, rating, genres, oscar_awards, cast, crew.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_name": {"type": "string", "description": "Movie name to search"}
                },
                "required": ["movie_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "movie_get_year_info",
            "description": "Get info about movies in a specific year (1990-2021). Returns movie list and oscar awards held that year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "string", "description": "Year string, e.g. '1992'"}
                },
                "required": ["year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "movie_get_movie_info_by_id",
            "description": "Get movie info by its unique ID. Returns same format as movie_get_movie_info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {"type": "integer", "description": "Unique movie ID"}
                },
                "required": ["movie_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "movie_get_person_info_by_id",
            "description": "Get person info by their unique ID. Returns same format as movie_get_person_info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "integer", "description": "Unique person ID"}
                },
                "required": ["person_id"]
            }
        }
    },
]


# =============================================================================
# FINANCE DOMAIN TOOLS
# =============================================================================

FINANCE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "finance_get_company_name",
            "description": "Search for company names matching a query. Returns a list of top matched company names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for company name"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finance_get_ticker_by_name",
            "description": "Get the ticker symbol for a given company name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Company name"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finance_get_price_history",
            "description": "Get 1-year daily price history (Open, Close, High, Low, Volume) for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_name": {"type": "string", "description": "Ticker symbol, e.g. 'META'"}
                },
                "required": ["ticker_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finance_get_detailed_price_history",
            "description": "Get past 5 days' 1-minute price history for a ticker (09:30-15:59 EST).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_name": {"type": "string", "description": "Ticker symbol"}
                },
                "required": ["ticker_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finance_get_dividends_history",
            "description": "Get dividend distribution history for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_name": {"type": "string", "description": "Ticker symbol"}
                },
                "required": ["ticker_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finance_get_market_capitalization",
            "description": "Get the market capitalization of a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_name": {"type": "string", "description": "Ticker symbol"}
                },
                "required": ["ticker_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finance_get_eps",
            "description": "Get earnings per share (EPS) for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_name": {"type": "string", "description": "Ticker symbol"}
                },
                "required": ["ticker_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finance_get_pe_ratio",
            "description": "Get the price-to-earnings (PE) ratio for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_name": {"type": "string", "description": "Ticker symbol"}
                },
                "required": ["ticker_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finance_get_info",
            "description": "Get meta information about a ticker (sector, industry, description, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_name": {"type": "string", "description": "Ticker symbol"}
                },
                "required": ["ticker_name"]
            }
        }
    },
]


# =============================================================================
# MUSIC DOMAIN TOOLS
# =============================================================================

MUSIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "music_search_artist_entity_by_name",
            "description": "Fuzzy search for music artists by name. Returns top-10 matching artist names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {"type": "string", "description": "Artist name to search"}
                },
                "required": ["artist_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_search_song_entity_by_name",
            "description": "Fuzzy search for songs by name. Returns top-10 matching song names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {"type": "string", "description": "Song name to search"}
                },
                "required": ["song_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_get_billboard_rank_date",
            "description": "Get the song and artist at a certain Billboard rank on a certain date. If no date given, returns all dates for that rank.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer", "description": "Billboard rank (1-100)"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format (optional)"}
                },
                "required": ["rank"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_get_billboard_attributes",
            "description": "Get attributes of a song from Billboard rankings on a certain date. Attributes: rank_last_week, weeks_in_chart, top_position, rank.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "attribute": {"type": "string", "description": "One of: rank_last_week, weeks_in_chart, top_position, rank"},
                    "song_name": {"type": "string", "description": "Song name"}
                },
                "required": ["date", "attribute", "song_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_grammy_get_best_artist_by_year",
            "description": "Get the Grammy Best New Artist winner for a specific year (1958-2019).",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Year in YYYY format"}
                },
                "required": ["year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_grammy_get_award_count_by_artist",
            "description": "Get total Grammy awards won by an artist (1958-2019).",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {"type": "string", "description": "Artist name"}
                },
                "required": ["artist_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_grammy_get_award_count_by_song",
            "description": "Get total Grammy awards won by a song (1958-2019).",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {"type": "string", "description": "Song name"}
                },
                "required": ["song_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_grammy_get_best_song_by_year",
            "description": "Get the Grammy Song of the Year for a specific year (1958-2019).",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Year in YYYY format"}
                },
                "required": ["year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_grammy_get_award_date_by_artist",
            "description": "Get the years an artist won Grammy awards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {"type": "string", "description": "Artist name"}
                },
                "required": ["artist_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_grammy_get_best_album_by_year",
            "description": "Get the Grammy Album of the Year for a specific year (1958-2019).",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Year in YYYY format"}
                },
                "required": ["year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_grammy_get_all_awarded_artists",
            "description": "Get all artists ever awarded Grammy Best New Artist (1958-2019).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_get_artist_birth_place",
            "description": "Get the birth place (2-digit country code, ISO-3166) of an artist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {"type": "string", "description": "Artist name"}
                },
                "required": ["artist_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_get_artist_birth_date",
            "description": "Get the birth date of an artist (or begin date of a band).",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {"type": "string", "description": "Artist name"}
                },
                "required": ["artist_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_get_members",
            "description": "Get the member list of a band.",
            "parameters": {
                "type": "object",
                "properties": {
                    "band_name": {"type": "string", "description": "Band name"}
                },
                "required": ["band_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_get_lifespan",
            "description": "Get the lifespan (birth and death dates) of an artist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {"type": "string", "description": "Artist name"}
                },
                "required": ["artist_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_get_song_author",
            "description": "Get the author of a song.",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {"type": "string", "description": "Song name"}
                },
                "required": ["song_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_get_song_release_country",
            "description": "Get the release country (2-digit ISO-3166 code) of a song.",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {"type": "string", "description": "Song name"}
                },
                "required": ["song_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_get_song_release_date",
            "description": "Get the release date of a song (YYYY-MM-DD format).",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {"type": "string", "description": "Song name"}
                },
                "required": ["song_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_get_artist_all_works",
            "description": "Get all works (songs/albums) by an artist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {"type": "string", "description": "Artist name"}
                },
                "required": ["artist_name"]
            }
        }
    },
]


# =============================================================================
# SPORTS DOMAIN TOOLS
# =============================================================================

SPORTS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sports_soccer_get_games_on_date",
            "description": "Get soccer games on a specific date. Returns venue, result, goals, opponent, captain info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD, YYYY-MM, or YYYY format"},
                    "team_name": {"type": "string", "description": "Team name (optional)"}
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sports_nba_get_games_on_date",
            "description": "Get NBA games on a specific date. Returns game_id, teams, win/loss, points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD, YYYY-MM, or YYYY format"},
                    "team_name": {"type": "string", "description": "Team name (optional), e.g. 'Los Angeles Lakers'"}
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sports_nba_get_play_by_play_data_by_game_ids",
            "description": "Get NBA play-by-play data for given game IDs. Returns event details, times, player info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "game_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of NBA game IDs, e.g. ['0022200547']"
                    }
                },
                "required": ["game_ids"]
            }
        }
    },
]


# =============================================================================
# TOOL REGISTRY
# =============================================================================

DOMAIN_TOOLS: Dict[str, List[dict]] = {
    "finance": FINANCE_TOOLS,
    "movie": MOVIE_TOOLS,
    "music": MUSIC_TOOLS,
    "sports": SPORTS_TOOLS,
    "open": OPEN_TOOLS,
}


def get_tools_for_domain(domain: str) -> List[dict]:
    """Return tools for a given domain + the universal answer tool."""
    domain_specific = DOMAIN_TOOLS.get(domain, OPEN_TOOLS)
    return domain_specific + [ANSWER_TOOL]


def get_all_tools() -> List[dict]:
    """Return all tools across all domains + answer tool."""
    all_tools = []
    for tools in DOMAIN_TOOLS.values():
        all_tools.extend(tools)
    all_tools.append(ANSWER_TOOL)
    return all_tools


# =============================================================================
# TOOL EXECUTOR
# =============================================================================

MAX_RESULT_LENGTH = 4000


class ToolExecutor:
    """Executes tool calls by delegating to the CRAG mock_api client."""

    def __init__(self, mock_api_url: str):
        self.api = CRAG(server=mock_api_url)

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return result as JSON string."""
        if tool_name == "answer":
            return arguments.get("answer", "")

        try:
            result = self._dispatch(tool_name, arguments)
            result_str = json.dumps(result, ensure_ascii=False, default=str)
            if len(result_str) > MAX_RESULT_LENGTH:
                result_str = result_str[:MAX_RESULT_LENGTH] + "... [truncated]"
            return result_str
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _dispatch(self, tool_name: str, args: dict) -> Any:
        """Route tool_name to the corresponding CRAG client method."""
        # Open domain
        if tool_name == "open_search_entity_by_name":
            return self.api.open_search_entity_by_name(args["query"])
        elif tool_name == "open_get_entity":
            return self.api.open_get_entity(args["entity"])

        # Movie domain
        elif tool_name == "movie_get_person_info":
            return self.api.movie_get_person_info(args["person_name"])
        elif tool_name == "movie_get_movie_info":
            return self.api.movie_get_movie_info(args["movie_name"])
        elif tool_name == "movie_get_year_info":
            return self.api.movie_get_year_info(args["year"])
        elif tool_name == "movie_get_movie_info_by_id":
            return self.api.movie_get_movie_info_by_id(int(args["movie_id"]))
        elif tool_name == "movie_get_person_info_by_id":
            return self.api.movie_get_person_info_by_id(int(args["person_id"]))

        # Finance domain
        elif tool_name == "finance_get_company_name":
            return self.api.finance_get_company_name(args["query"])
        elif tool_name == "finance_get_ticker_by_name":
            return self.api.finance_get_ticker_by_name(args["query"])
        elif tool_name == "finance_get_price_history":
            return self.api.finance_get_price_history(args["ticker_name"])
        elif tool_name == "finance_get_detailed_price_history":
            return self.api.finance_get_detailed_price_history(args["ticker_name"])
        elif tool_name == "finance_get_dividends_history":
            return self.api.finance_get_dividends_history(args["ticker_name"])
        elif tool_name == "finance_get_market_capitalization":
            return self.api.finance_get_market_capitalization(args["ticker_name"])
        elif tool_name == "finance_get_eps":
            return self.api.finance_get_eps(args["ticker_name"])
        elif tool_name == "finance_get_pe_ratio":
            return self.api.finance_get_pe_ratio(args["ticker_name"])
        elif tool_name == "finance_get_info":
            return self.api.finance_get_info(args["ticker_name"])

        # Music domain
        elif tool_name == "music_search_artist_entity_by_name":
            return self.api.music_search_artist_entity_by_name(args["artist_name"])
        elif tool_name == "music_search_song_entity_by_name":
            return self.api.music_search_song_entity_by_name(args["song_name"])
        elif tool_name == "music_get_billboard_rank_date":
            return self.api.music_get_billboard_rank_date(
                int(args["rank"]), args.get("date")
            )
        elif tool_name == "music_get_billboard_attributes":
            return self.api.music_get_billboard_attributes(
                args["date"], args["attribute"], args["song_name"]
            )
        elif tool_name == "music_grammy_get_best_artist_by_year":
            return self.api.music_grammy_get_best_artist_by_year(int(args["year"]))
        elif tool_name == "music_grammy_get_award_count_by_artist":
            return self.api.music_grammy_get_award_count_by_artist(args["artist_name"])
        elif tool_name == "music_grammy_get_award_count_by_song":
            return self.api.music_grammy_get_award_count_by_song(args["song_name"])
        elif tool_name == "music_grammy_get_best_song_by_year":
            return self.api.music_grammy_get_best_song_by_year(int(args["year"]))
        elif tool_name == "music_grammy_get_award_date_by_artist":
            return self.api.music_grammy_get_award_date_by_artist(args["artist_name"])
        elif tool_name == "music_grammy_get_best_album_by_year":
            return self.api.music_grammy_get_best_album_by_year(int(args["year"]))
        elif tool_name == "music_grammy_get_all_awarded_artists":
            return self.api.music_grammy_get_all_awarded_artists()
        elif tool_name == "music_get_artist_birth_place":
            return self.api.music_get_artist_birth_place(args["artist_name"])
        elif tool_name == "music_get_artist_birth_date":
            return self.api.music_get_artist_birth_date(args["artist_name"])
        elif tool_name == "music_get_members":
            return self.api.music_get_members(args["band_name"])
        elif tool_name == "music_get_lifespan":
            return self.api.music_get_lifespan(args["artist_name"])
        elif tool_name == "music_get_song_author":
            return self.api.music_get_song_author(args["song_name"])
        elif tool_name == "music_get_song_release_country":
            return self.api.music_get_song_release_country(args["song_name"])
        elif tool_name == "music_get_song_release_date":
            return self.api.music_get_song_release_date(args["song_name"])
        elif tool_name == "music_get_artist_all_works":
            return self.api.music_get_artist_all_works(args["artist_name"])

        # Sports domain
        elif tool_name == "sports_soccer_get_games_on_date":
            return self.api.sports_soccer_get_games_on_date(
                date=args["date"], team_name=args.get("team_name")
            )
        elif tool_name == "sports_nba_get_games_on_date":
            return self.api.sports_nba_get_games_on_date(
                date=args["date"], team_name=args.get("team_name")
            )
        elif tool_name == "sports_nba_get_play_by_play_data_by_game_ids":
            return self.api.sports_nba_get_play_by_play_data_by_game_ids(args["game_ids"])

        else:
            raise ValueError(f"Unknown tool: {tool_name}")


if __name__ == "__main__":
    # Verify tool counts
    for domain, tools in DOMAIN_TOOLS.items():
        print(f"{domain}: {len(tools)} tools")
    print(f"Total unique tools: {sum(len(t) for t in DOMAIN_TOOLS.values())} + 1 answer tool")
    print("ok")
