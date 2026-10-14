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

    cloud_client = QdrantClient(url=url, port=443, api_key=api_key, timeout=120)

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
        )
        from qdrant_client.models import PayloadSchemaType
        for f in ['chunk_id', 'parent_chunk_id', 'company', 'ticker', 'source_file', 'fiscal_year', 'section', 'subsection', 'content_type', 'chunk_level']:
            cloud_client.create_payload_index(collection_name=collection_name, field_name=f, field_schema=PayloadSchemaType.KEYWORD)
        cloud_client.create_payload_index(collection_name=collection_name, field_name='page_number', field_schema=PayloadSchemaType.INTEGER)

    # 4. Fetch existing points in Cloud to resume without re-uploading
    print("Checking existing points in Cloud for fast resume...")
    existing_ids = set()
    cloud_offset = None
    while True:
        records, next_c_offset = cloud_client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=cloud_offset,
            with_payload=False,
            with_vectors=False
        )
        for r in records:
            existing_ids.add(r.id)
        cloud_offset = next_c_offset
        if cloud_offset is None or not records:
            break

    print(f"Found {len(existing_ids)} points already migrated. Only uploading remaining points...")

    # 5. Migrate remaining points in batches
    print("Starting migration...")
    import time
    batch_size = 25
    offset = None
    total_migrated = len(existing_ids)

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

        # Filter out points that already exist in cloud
        point_structs = [
            PointStruct(id=record.id, vector=record.vector, payload=record.payload)
            for record in records
            if record.id not in existing_ids
        ]

        if point_structs:
            # Upsert to cloud with retries
            for attempt in range(1, 5):
                try:
                    cloud_client.upsert(
                        collection_name=collection_name,
                        points=point_structs,
                        wait=True
                    )
                    break
                except Exception as e:
                    if attempt == 4:
                        raise e
                    print(f"Batch write retry {attempt}/4 due to: {e}... waiting 3s")
                    time.sleep(3)

            total_migrated += len(point_structs)
            print(f"Progress: {total_migrated}/5180 points...")

        offset = next_offset
        if offset is None:
            break

    print(f"\nMigration complete! All {total_migrated} vectors are live on Qdrant Cloud.")

if __name__ == "__main__":
    migrate()
