import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from qdrant_client.models import PointStruct

def migrate():
    # Load .env manually in case this is run directly
    load_dotenv()

    # 1. Local Client
    local_path = "./data/qdrant_store"
    if not os.path.exists(local_path):
        print(f"Error: Local Qdrant store not found at {local_path}")
        return

    local_client = QdrantClient(path=local_path)
    collection_name = "financial_rag"

    # 2. Cloud Client
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")

    if not url or not api_key:
        print("Error: QDRANT_URL or QDRANT_API_KEY not found in environment.")
        print("Please add them to your .env file.")
        return

    cloud_client = QdrantClient(url=url, port=443, api_key=api_key, timeout=60)

    print(f"Connected to Local ({local_path}) and Cloud ({url})")

    # 3. Create Cloud Collection if missing
    try:
        local_info = local_client.get_collection(collection_name)
    except Exception as e:
        print(f"Error reading local collection: {e}")
        return

    existing_cloud = [c.name for c in cloud_client.get_collections().collections]
    if collection_name not in existing_cloud:
        print(f"Creating collection '{collection_name}' in Cloud...")
        cloud_client.create_collection(
            collection_name=collection_name,
            vectors_config=local_info.config.params.vectors,
            hnsw_config=local_info.config.hnsw_config.model_dump() if hasattr(local_info.config.hnsw_config, 'model_dump') else local_info.config.hnsw_config.dict()
        )

    # 4. Migrate points in batches
    print("Starting migration...")
    batch_size = 100
    offset = None
    total_migrated = 0

    while True:
        # Scroll through local points
        records, next_offset = local_client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )

        if not records:
            break

        # Convert Record objects to PointStruct objects
        point_structs = [
            PointStruct(id=record.id, vector=record.vector, payload=record.payload)
            for record in records
        ]

        # Upsert to cloud
        cloud_client.upsert(
            collection_name=collection_name,
            points=point_structs
        )

        total_migrated += len(records)
        print(f"Migrated {total_migrated} points...")

        offset = next_offset
        if offset is None:
            break

    print(f"\nMigration complete! Successfully moved {total_migrated} vectors to Qdrant Cloud.")

if __name__ == "__main__":
    migrate()
