import os
import json
import hashlib
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import JSONLoader
from src.models import PineconeRecord, SportsMetadata, PerformanceStat, GameContext

load_dotenv(find_dotenv())

def clean_display_value(val: str) -> str:
    """Removes messy ratios like '354/492' and keeps core stats."""
    if not val: return "N/A"
    parts = val.split(',')
    cleaned = [p.strip() for p in parts if '/' not in p]
    return ", ".join(cleaned) if cleaned else val

def extract_leaders(comp: dict) -> list:
    """Aggregates leaders from both Competition (Game) and Competitor (Team) levels."""
    all_leaders = []
    
    if "leaders" in comp:
        for cat in comp["leaders"]:
            all_leaders.append({
                "category": cat["displayName"],
                "data": cat["leaders"][0] if cat.get("leaders") else None,
                "team": None
            })

    for team_entry in comp.get("competitors", []):
        team_name = team_entry["team"]["abbreviation"]
        for cat in team_entry.get("leaders", []):
            all_leaders.append({
                "category": f"{team_name} {cat['displayName']}",
                "data": cat["leaders"][0] if cat.get("leaders") else None,
                "team": team_name
            })
            
    return [l for l in all_leaders if l["data"]]

def extract_team_stats(comp: dict) -> str:
    """Extracts key team stats like FG% or 3P%."""
    stats_summary = []
    for c in comp.get("competitors", []):
        abbr = c["team"]["abbreviation"]
        key_stats = []
        for stat in c.get("statistics", []):
            if stat["name"] in ["fieldGoalPct", "threePointPct", "freeThrowPct"]:
                key_stats.append(f"{stat['abbreviation']}: {stat['displayValue']}%")
        if key_stats:
            stats_summary.append(f"{abbr} [{', '.join(key_stats)}]")
    return " | ".join(stats_summary)

def transform_event_to_text(event: dict) -> str:
    """Creates a rich, legible Knowledge Card."""
    comp = event.get("competitions", [{}])[0]
    status_obj = event.get("status", {}).get("type", {})
    state = status_obj.get("state", "pre") # pre, in, post
    detail = status_obj.get("detail", "TBD")
    matchup = event.get("name", "Unknown Matchup")
    
    score_line = " vs ".join([f"{c['team']['displayName']} ({c.get('score','0')})" for c in comp.get("competitors", [])])
    
    leaders = extract_leaders(comp)
    leader_text_list = []
    for l in leaders:
        name = l['data'].get('athlete', {}).get('displayName', 'Unknown')
        val = clean_display_value(l['data'].get('displayValue', ''))
        leader_text_list.append(f"{l['category']}: {name} ({val})")
    leader_text = " | ".join(leader_text_list)

    team_stats_text = extract_team_stats(comp)
    stats_block = f"TEAM STATS: {team_stats_text}. " if team_stats_text else ""

    venue = comp.get("venue", {}).get("fullName", "TBD")
    odds = comp.get("odds", [{}])[0].get("details", "N/A") if comp.get("odds") else "N/A"

    if state == "pre":
        return (f"PREVIEW: {matchup} at {venue}. TIME: {event.get('date')}. "
                f"ODDS: {odds}. "
                f"PROJECTED LEADERS: {leader_text}.")
    elif state == "in":
        return (f"LIVE GAME ({detail}): {score_line}. "
                f"{stats_block}"
                f"LEADERS: {leader_text}.")
    else: 
        return (f"FINAL SCORE: {score_line}. "
                f"{stats_block}"
                f"TOP PERFORMERS: {leader_text}.")

def extract_smart_metadata(raw_data: dict, filename: str, stable_id: str) -> SportsMetadata:
    is_news = "news" in filename
    sport = "nba" if "nba" in filename else "nfl"
    
    teams, athletes, performers, score_list = [], [], [], []
    context, team_stats_str = None, None

    # HASH is still useful for deduplication checking, but NOT for the ID
    c_hash = hashlib.md5(json.dumps(raw_data, sort_keys=True).encode('utf-8')).hexdigest()

    if not is_news:
        comp = raw_data.get("competitions", [{}])[0]
        status_obj = raw_data.get("status", {}).get("type", {})
        
        for c in comp.get("competitors", []):
            teams.append(c["team"]["displayName"])
            score_list.append(f"{c['team']['displayName']} {c.get('score', '0')}")

        raw_leaders = extract_leaders(comp)
        for l in raw_leaders:
            name = l['data'].get('athlete', {}).get('displayName', 'Unknown')
            val = clean_display_value(l['data'].get('displayValue', ''))
            athletes.append(name)
            performers.append(PerformanceStat(
                category=l['category'], 
                athlete_name=name, 
                display_value=val,
                team=l['team']
            ))

        weather = raw_data.get("weather", {}).get("displayValue", "")
        odds_list = comp.get("odds", [])
        odds = odds_list[0].get("details") if odds_list else "N/A"
        context = GameContext(
            venue=comp.get("venue", {}).get("fullName"),
            location=comp.get("venue", {}).get("address", {}).get("city"),
            weather=weather,
            odds=odds,
            broadcast=raw_data.get("broadcast")
        )
        team_stats_str = extract_team_stats(comp)

        headline = raw_data.get("name")
        event_date = raw_data.get("date")
        status = status_obj.get("state") 

    else:
        headline = raw_data.get("headline")
        event_date = raw_data.get("published")
        status = "news"
        for cat in raw_data.get("categories", []):
            if cat.get("type") == "team": teams.append(cat.get("description"))
            elif cat.get("type") == "athlete": athletes.append(cat.get("description"))

    return SportsMetadata(
        sport=sport, content_type="news" if is_news else "score",
        content_hash=c_hash, source_file=filename,
        teams=list(set(teams)), athletes=list(set(athletes)),
        headline=headline, event_date=event_date, status=status,
        performers=performers, context=context, scores=score_list,
        team_stats=team_stats_str
    )

def run_ingestion():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("sports-data-index")

    file_schema_map = {
        "nba-news.json": ".articles[]?",
        "nfl-news.json": ".articles[]?",
        "nba-score.json": ".events[]?",
        "nfl-score.json": ".events[]?"
    }
    
    all_records = []
    for filename, schema in file_schema_map.items():
        file_path = f"data/{filename}"
        if not os.path.exists(file_path): continue
            
        print(f"Processing {filename}...")
        loader = JSONLoader(file_path=file_path, jq_schema=schema, text_content=False)
        docs = loader.load()
        
        for doc in docs:
            raw_json = json.loads(doc.page_content)
            
            # --- STABLE ID LOGIC ---
            # Use the actual Game ID or Article ID from ESPN
            # This ensures 'Patriots vs Broncos' always has the SAME Pinecone ID
            # regardless of whether it's Scheduled or Final.
            entity_id = raw_json.get("id")
            if not entity_id:
                # Fallback if no ID exists
                entity_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            
            stable_id = f"doc-{entity_id}"

            smart_meta = extract_smart_metadata(raw_json, filename, stable_id)
            
            if smart_meta.content_type == "news":
                chunk_text = f"NEWS: {smart_meta.headline}. {raw_json.get('description', '')}"
            else:
                chunk_text = transform_event_to_text(raw_json)
            
            meta_dict = smart_meta.model_dump(exclude_none=True)
            if "performers" in meta_dict: 
                meta_dict["performers"] = json.dumps([p.model_dump() for p in smart_meta.performers])
            if "context" in meta_dict and smart_meta.context: 
                meta_dict["context"] = json.dumps(smart_meta.context.model_dump())

            all_records.append({
                "id": stable_id, 
                "chunk_text": chunk_text, 
                **meta_dict
            })

    if all_records:
        # Deduplicate locally
        unique_records = list({r['id']: r for r in all_records}.values())
        print(f"Upserting {len(unique_records)} records...")
        index.upsert_records(namespace="sports-info", records=unique_records)
        print("Ingestion Complete!")

if __name__ == "__main__":
    run_ingestion()