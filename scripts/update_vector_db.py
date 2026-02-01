import os
import httpx
import asyncio
import hashlib
from typing import List
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from models import SportsMetadata, PerformanceStat, PineconeRecord, SearchOptimization

load_dotenv()

class SportsDataSync:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index = self.pc.Index("sports-data-index")

    async def fetch_api(self, sport: str, league: str, type: str):
        url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/{type}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            return resp.json(), sport, league, type

    def generate_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_embedding(self, text: str) -> List[float]:
        return self.openai.embeddings.create(
            input=[text.replace("\n", " ")], 
            model="text-embedding-3-small",
            dimensions=1024
        ).data[0].embedding

    def process_news(self, data, sport, league) -> List[PineconeRecord]:
        records = []
        for art in data.get("articles", []):
            # Map athletes and teams for search optimization
            athletes = [cat.get("description") for cat in art.get("categories", []) if cat.get("type") == "athlete"]
            teams = [cat.get("description") for cat in art.get("categories", []) if cat.get("type") == "team"]
            
            meta = SportsMetadata(
                sport=sport,
                content_type="news",
                content_hash=self.generate_hash(art['headline']),
                source_file=f"{sport}-{league}-news",
                headline=art['headline'],
                summary=art.get('description'),
                event_date=art.get('published'),
                athletes=athletes,
                teams=teams,
                discovery=SearchOptimization(
                    keywords=athletes + teams + [art['headline']],
                    hashtags=[f"#{t.replace(' ', '')}" for t in teams],
                    social_handles=[cat.get("guid") for cat in art.get("categories", []) if cat.get("type") == "guid"]
                )
            )
            records.append(PineconeRecord(id=f"news-{art['id']}", chunk_text=art['headline'], metadata=meta))
        return records

    def process_scores(self, data, sport, league) -> List[PineconeRecord]:
        records = []
        for ev in data.get("events", []):
            comp = ev['competitions'][0]
            scores = [f"{t['team']['displayName']} {t['score']}" for t in comp['competitors']]
            
            meta = SportsMetadata(
                sport=sport,
                content_type="score",
                content_hash=self.generate_hash(ev['name']),
                source_file=f"{sport}-{league}-scores",
                headline=ev['name'],
                status=ev['status']['type']['state'],
                event_date=ev['date'],
                scores=scores,
                teams=[t['team']['displayName'] for t in comp['competitors']],
                discovery=SearchOptimization(
                    keywords=[ev['name']] + scores,
                    suggested_queries=[f"{ev['name']} latest score", f"{ev['name']} highlights"]
                )
            )
            records.append(PineconeRecord(id=f"score-{ev['id']}", chunk_text=ev['name'], metadata=meta))
        return records

    async def sync(self):
        tasks = [
            self.fetch_api("football", "nfl", "news"),
            self.fetch_api("basketball", "nba", "news"),
            self.fetch_api("football", "nfl", "scoreboard"),
            self.fetch_api("basketball", "nba", "scoreboard")
        ]
        results = await asyncio.gather(*tasks)
        
        all_records = []
        for data, sport, league, dtype in results:
            if dtype == "news":
                all_records.extend(self.process_news(data, sport, league))
            else:
                all_records.extend(self.process_scores(data, sport, league))

        # Vectorize and Upsert
        upserts = []
        for rec in all_records:
            vector = self.get_embedding(f"{rec.metadata.headline}: {rec.metadata.summary or ''}")
            upserts.append({
                "id": rec.id, 
                "values": vector, 
                "metadata": rec.metadata.model_dump(exclude_none=True)
            })

        if upserts:
            self.index.upsert(vectors=upserts)
            print(f"Sync Complete: {len(upserts)} items updated.")

if __name__ == "__main__":
    sync_manager = SportsDataSync()
    asyncio.run(sync_manager.sync())