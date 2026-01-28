import os
import json
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone

load_dotenv(find_dotenv())

class SportsRetriever:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in .env")
        
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = "sports-data-index"
        self.index = self.pc.Index(self.index_name)
        self.namespace = "sports-info"

    def search(self, query: str, sport: str = None, top_k: int = 3):
        """
        Searches Pinecone for relevant sports context.
        """
        # 1. Construct the Search Payload
        search_query = {
            "inputs": {"text": query},
            "top_k": top_k
        }

        # 2. Add Filter directly to the query object (The Fix)
        if sport:
            search_query["filter"] = {"sport": sport.lower()}

        try:
            # 3. Execute Search (Removed 'query_parameters')
            response = self.index.search_records(
                namespace=self.namespace,
                query=search_query
            )
            
            results = []
            for hit in response.get("result", {}).get("hits", []):
                fields = hit.get("fields", {})
                
                # Deserializing stored JSON strings
                performers = []
                if "performers" in fields:
                    try:
                        performers = json.loads(fields["performers"])
                    except:
                        pass

                context = {}
                if "context" in fields:
                    try:
                        context = json.loads(fields["context"])
                    except:
                        pass

                results.append({
                    "score": hit.get("_score"),
                    "text": fields.get("chunk_text"),
                    "headline": fields.get("headline"),
                    "date": fields.get("event_date"),
                    "performers": performers,
                    "context": context
                })
                
            return results

        except Exception as e:
            print(f"❌ Retrieval Error: {e}")
            return []