from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PerformanceStat(BaseModel):
    category: str        # e.g., "Passing Yards", "Points"
    athlete_name: str
    display_value: str   # Cleaned value (e.g., "4394 YDS")
    team: Optional[str] = None

class GameContext(BaseModel):
    venue: Optional[str] = None
    location: Optional[str] = None
    weather: Optional[str] = None
    odds: Optional[str] = None
    broadcast: Optional[str] = None

class SportsMetadata(BaseModel):
    # System Fields
    sport: str             # "nba" or "nfl"
    content_type: str      # "news" or "score"
    content_hash: str
    source_file: str

    # Team & Athlete Indexing
    teams: List[str] = Field(default_factory=list)
    athletes: List[str] = Field(default_factory=list)
    
    # Event Info
    headline: Optional[str] = None
    event_date: Optional[str] = None
    status: Optional[str] = None # "pre", "in", "post"
    
    # Rich Data
    scores: List[str] = Field(default_factory=list) # ["Lakers 110", "Kings 98"]
    performers: List[PerformanceStat] = Field(default_factory=list)
    context: Optional[GameContext] = None
    team_stats: Optional[str] = None # e.g. "NYK: 50% FG | SAC: 43% FG"

class PineconeRecord(BaseModel):
    id: str
    chunk_text: str
    metadata: SportsMetadata