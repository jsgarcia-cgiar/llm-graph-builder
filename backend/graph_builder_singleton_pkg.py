import os
import json
import argparse
import logging
import asyncio
from typing import Optional, Tuple, Dict, Any, List
import hashlib

# Import from installed package
from llm_graph_builder import (
    processing_source,
    get_documents_from_s3,
    create_graph_database_connection,
    graphDBdataAccess,
    sourceNode,
)
from datetime import datetime


class GraphBuilderSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GraphBuilderSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
    ):
        # Load from env if not provided
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME", "")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        # Minimal defaults for chunking and processing
        self.default_token_chunk_size = int(os.getenv("TOKEN_CHUNK_SIZE", 2000))
        self.default_chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        self.default_chunks_to_combine = int(os.getenv("CHUNKS_TO_COMBINE", 4))

        # Controls batch upserts inside processing_source (uses env UPDATE_GRAPH_CHUNKS_PROCESSED)
        os.environ.setdefault("UPDATE_GRAPH_CHUNKS_PROCESSED", "5")
        os.environ.setdefault("IS_EMBEDDING", "TRUE")

        # Logging setup
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
        )

    def _validate_neo4j(self):
        missing = [
            k
            for k, v in {
                "NEO4J_URI": self.neo4j_uri,
                "NEO4J_USERNAME": self.neo4j_username,
                "NEO4J_PASSWORD": self.neo4j_password,
            }.items()
            if not v
        ]
        if missing:
            raise ValueError(f"Missing Neo4j config for: {', '.join(missing)}")

    def _maybe_set_s3_endpoint(self, endpoint_url: Optional[str]):
        # Backblaze B2 S3-compatible endpoint support
        if endpoint_url:
            os.environ["AWS_ENDPOINT_URL_S3"] = endpoint_url
            # boto3 respects AWS_ENDPOINT_URL or per-service AWS_ENDPOINT_URL_S3
            os.environ.setdefault("AWS_ENDPOINT_URL", endpoint_url)

    def _ensure_document_node(
        self,
        file_name: str,
        model: str,
        source: str,
        url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
    ) -> None:
        graph = create_graph_database_connection(
            self.neo4j_uri,
            self.neo4j_username,
            self.neo4j_password,
            self.neo4j_database,
        )
        graph_access = graphDBdataAccess(graph)
        existing = graph_access.get_current_status_document_node(file_name)
        if not existing:
            obj = sourceNode()
            obj.file_name = file_name
            obj.file_type = "text"
            obj.file_size = 0
            obj.file_source = source
            obj.model = model
            obj.url = url
            obj.awsAccessKeyId = aws_access_key_id
            obj.created_at = datetime.now()
            obj.chunkNodeCount = 0
            obj.chunkRelCount = 0
            obj.entityNodeCount = 0
            obj.entityEntityRelCount = 0
            obj.communityNodeCount = 0
            obj.communityRelCount = 0
            graph_access.create_source_node(obj)

    def _upsert_chunks(
        self,
        file_name: str,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """Insert/merge provided chunks into Neo4j for given file_name.

        chunks: list of {"text": str, optional: "position": int, "page_number": int,
                         "start_time": float, "end_time": float}
        Returns number of chunks processed.
        """
        if not chunks:
            raise RuntimeError("No chunks provided to upsert")

        # Sort by provided position if present; otherwise keep given order
        if all(isinstance(c.get("position"), int) for c in chunks if "position" in c):
            chunks_sorted = sorted(chunks, key=lambda c: c.get("position", 0))
        else:
            chunks_sorted = list(chunks)

        graph = create_graph_database_connection(
            self.neo4j_uri,
            self.neo4j_username,
            self.neo4j_password,
            self.neo4j_database,
        )
        graph_access = graphDBdataAccess(graph)

        batch_data: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        current_chunk_id = ""
        offset = 0
        for i, chunk in enumerate(chunks_sorted):
            text = str(chunk.get("text", ""))
            if not text:
                # skip empty chunks
                continue
            previous_chunk_id = current_chunk_id
            current_chunk_id = hashlib.sha1(text.encode()).hexdigest()
            position = int(chunk.get("position", i + 1))
            if i > 0:
                prev_text = str(chunks_sorted[i - 1].get("text", ""))
                offset += len(prev_text)

            chunk_data: Dict[str, Any] = {
                "id": current_chunk_id,
                "pg_content": text,
                "position": position,
                "length": len(text),
                "f_name": file_name,
                "previous_id": previous_chunk_id,
                "content_offset": offset,
            }
            if "page_number" in chunk and chunk["page_number"] is not None:
                chunk_data["page_number"] = chunk["page_number"]
            if "start_time" in chunk and "end_time" in chunk:
                chunk_data["start_time"] = chunk["start_time"]
                chunk_data["end_time"] = chunk["end_time"]

            batch_data.append(chunk_data)

            # FIRST_CHUNK or NEXT_CHUNK relations blueprint
            if i == 0:
                relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
            else:
                relationships.append(
                    {
                        "type": "NEXT_CHUNK",
                        "previous_chunk_id": previous_chunk_id,
                        "current_chunk_id": current_chunk_id,
                    }
                )

        if not batch_data:
            return 0

        # Create/merge chunk nodes + PART_OF
        query_create_chunks = """
            UNWIND $batch_data AS data
            MERGE (c:Chunk {id: data.id})
            SET c.text = data.pg_content,
                c.position = data.position,
                c.length = data.length,
                c.fileName = data.f_name,
                c.content_offset = data.content_offset
            WITH data, c
            SET c.page_number = CASE WHEN data.page_number IS NOT NULL THEN data.page_number END,
                c.start_time = CASE WHEN data.start_time IS NOT NULL THEN data.start_time END,
                c.end_time = CASE WHEN data.end_time IS NOT NULL THEN data.end_time END
            WITH data, c
            MATCH (d:Document {fileName: data.f_name})
            MERGE (c)-[:PART_OF]->(d)
        """
        graph_access.execute_query(query_create_chunks, {"batch_data": batch_data})
        graph_access.execute_query(query_create_chunks, {"batch_data": batch_data})

        # FIRST_CHUNK
        query_first = """
            UNWIND $relationships AS relationship
            MATCH (d:Document {fileName: $f_name})
            MATCH (c:Chunk {id: relationship.chunk_id})
            FOREACH(_ IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                MERGE (d)-[:FIRST_CHUNK]->(c))
        """
        graph_access.execute_query(query_first, {"f_name": file_name, "relationships": relationships})
        graph_access.execute_query(query_first, {"f_name": file_name, "relationships": relationships})

        # NEXT_CHUNK
        query_next = """
            UNWIND $relationships AS relationship
            MATCH (c:Chunk {id: relationship.current_chunk_id})
            WITH c, relationship
            MATCH (pc:Chunk {id: relationship.previous_chunk_id})
            FOREACH(_ IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
        graph_access.execute_query(query_next, {"relationships": relationships})

        return len(batch_data)

    async def run_from_s3(
        self,
        s3_url: str,
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        b2_bucket: str,
        model: str,
        *,
        allowed_nodes: str = "",
        allowed_relationship: str = "",
        token_chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunks_to_combine: Optional[int] = None,
        backblaze_endpoint_url: Optional[str] = None,
        retry_condition: Optional[str] = None,
        additional_instructions: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run full workflow for a single S3 (Backblaze) object URL.

        s3_url: s3://bucket/path/to/file.pdf
        model: e.g. "openai-gpt-4o" (requires env LLM_MODEL_CONFIG_openai-gpt-4o)

        Returns: (uri_latency, response) from processing_source
        """
        self._validate_neo4j()
        self._maybe_set_s3_endpoint(backblaze_endpoint_url)

        # Fetch document pages using existing helper
        logging.info("Fetching document from S3-compatible storage")
        file_name, pages = get_documents_from_s3(
            s3_url,
            aws_access_key_id,
            aws_secret_access_key,
            endpoint_url=backblaze_endpoint_url,
            bucket=b2_bucket,
        )
        print(f"FILENAME: {file_name}\nPAGES: {len(pages)}")
        if not pages:
            raise RuntimeError(f"No content returned for {file_name}")

        # Ensure Document node exists before processing (mirrors /url/scan step)
        graph = create_graph_database_connection(
            self.neo4j_uri,
            self.neo4j_username,
            self.neo4j_password,
            self.neo4j_database,
        )
        graph_access = graphDBdataAccess(graph)
        existing = graph_access.get_current_status_document_node(file_name)
        if not existing:
            obj = sourceNode()
            obj.file_name = file_name
            obj.file_type = "pdf"
            obj.file_size = 0
            obj.file_source = "s3 bucket"
            obj.model = model
            obj.url = s3_url
            obj.awsAccessKeyId = aws_access_key_id
            obj.created_at = datetime.now()
            obj.chunkNodeCount = 0
            obj.chunkRelCount = 0
            obj.entityNodeCount = 0
            obj.entityEntityRelCount = 0
            obj.communityNodeCount = 0
            obj.communityRelCount = 0
            graph_access.create_source_node(obj)

        # Kick off processing using the same internal pipeline as the API
        logging.info("Starting processing_source workflow")
        uri_latency, response = await processing_source(
            self.neo4j_uri,
            self.neo4j_username,
            self.neo4j_password,
            self.neo4j_database,
            model,
            file_name,
            pages,
            allowed_nodes or "",
            allowed_relationship or "",
            token_chunk_size or self.default_token_chunk_size,
            chunk_overlap or self.default_chunk_overlap,
            chunks_to_combine or self.default_chunks_to_combine,
            additional_instructions=additional_instructions,
        )

        # Update node/relationship counts similar to /extract endpoint
        graph = create_graph_database_connection(
            self.neo4j_uri,
            self.neo4j_username,
            self.neo4j_password,
            self.neo4j_database,
        )
        graph_access = graphDBdataAccess(graph)
        graph_access.update_node_relationship_count(file_name)
        return uri_latency, response

    async def run_with_external_chunks(
        self,
        *,
        model: str,
        file_name: str,
        chunks: List[Dict[str, Any]],
        allowed_nodes: str = "",
        allowed_relationship: str = "",
        additional_instructions: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run workflow using caller-provided chunks (e.g., from Qdrant).

        This will:
        - Ensure Document node exists
        - Upsert provided chunks as Chunk nodes linked to the Document
        - Invoke processing in retry mode so the pipeline reuses existing chunks
        """
        self._validate_neo4j()

        # Ensure Document presence
        self._ensure_document_node(file_name, model, source="external-chunks")

        # Upsert chunks
        logging.info("Upserting external chunks into Neo4j")
        inserted = self._upsert_chunks(file_name, chunks)
        if inserted == 0:
            raise RuntimeError("No valid chunks to insert")

        # Now run pipeline in retry mode to reuse existing chunks
        logging.info("Starting processing_source workflow (retry=start_from_beginning)")
        uri_latency, response = await processing_source(
            self.neo4j_uri,
            self.neo4j_username,
            self.neo4j_password,
            self.neo4j_database,
            model,
            file_name,
            [],
            allowed_nodes or "",
            allowed_relationship or "",
            self.default_token_chunk_size,
            self.default_chunk_overlap,
            self.default_chunks_to_combine,
            retry_condition="start_from_beginning",
            additional_instructions=additional_instructions,
        )

        # Update counts
        graph = create_graph_database_connection(
            self.neo4j_uri,
            self.neo4j_username,
            self.neo4j_password,
            self.neo4j_database,
        )
        graph_access = graphDBdataAccess(graph)
        graph_access.update_node_relationship_count(file_name)
        return uri_latency, response


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Run LLM Graph Builder pipeline on a single S3/Backblaze file"
    )
    parser.add_argument(
        "--s3-url",
        required=False,
        help="s3://bucket/path/to/file.pdf (Backblaze S3-compatible works)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model key, e.g. openai-gpt-4o, gemini-1.5-pro, diffbot",
    )
    parser.add_argument(
        "--aws-access-key-id", default=os.getenv("AWS_ACCESS_KEY_ID", "")
    )
    parser.add_argument(
        "--aws-secret-access-key", default=os.getenv("AWS_SECRET_ACCESS_KEY", "")
    )
    parser.add_argument(
        "--b2-endpoint",
        default=os.getenv("AWS_ENDPOINT_URL_S3", ""),
        help="Backblaze S3 endpoint, e.g. https://s3.us-west-000.backblazeb2.com",
    )
    parser.add_argument(
        "--allowed-nodes",
        default=os.getenv("ALLOWED_NODES", ""),
        help="Comma-separated node labels",
    )
    parser.add_argument(
        "--allowed-relationship",
        default=os.getenv("ALLOWED_RELATIONSHIP", ""),
        help="Comma-separated triplets: src,REL,tgt,src,REL,tgt...",
    )
    parser.add_argument(
        "--token-chunk-size", type=int, default=int(os.getenv("TOKEN_CHUNK_SIZE", 2000))
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", 200))
    )
    parser.add_argument(
        "--chunks-to-combine", type=int, default=int(os.getenv("CHUNKS_TO_COMBINE", 4))
    )
    parser.add_argument(
        "--additional-instructions", default=os.getenv("ADDITIONAL_INSTRUCTIONS", "")
    )
    parser.add_argument("--b2-bucket", default="ask-papa-ai-dx--dev--private")
    parser.add_argument("--chunks-json", default="", help="Path to JSON file containing chunks: [{text, position?, page_number?}] for external processing")
    parser.add_argument("--file-name", default="", help="Logical document name when using external chunks")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", ""), help="Qdrant URL, e.g. https://qdrant.local:6333")
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", ""), help="Qdrant API key if required")
    parser.add_argument("--qdrant-collection", default=os.getenv("QDRANT_COLLECTION", ""), help="Qdrant collection name")
    parser.add_argument("--qdrant-doc-key", default=os.getenv("QDRANT_DOC_KEY", "file_name"), help="Payload key that identifies the document")
    parser.add_argument("--qdrant-doc-value", default="", help="Payload value to select chunks for this document")
    parser.add_argument("--qdrant-text-key", default=os.getenv("QDRANT_TEXT_KEY", "text"), help="Payload key holding chunk text")
    parser.add_argument("--qdrant-order-key", default=os.getenv("QDRANT_ORDER_KEY", "position"), help="Payload key holding chunk order/position")
    parser.add_argument("--qdrant-page-key", default=os.getenv("QDRANT_PAGE_KEY", "page_number"), help="Payload key holding page number")
    args = parser.parse_args()

    # Validate required env for LLM
    env_key = f"LLM_MODEL_CONFIG_{args.model.lower().strip()}"
    if not os.getenv(env_key):
        raise SystemExit(
            f"Missing required env: {env_key}. See backend/example.env for format."
        )

    # Validate Neo4j basics
    neo4j_uri = os.getenv("NEO4J_URI", "")
    neo4j_user = os.getenv("NEO4J_USERNAME", "")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "")
    neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
    if not (neo4j_uri and neo4j_user and neo4j_pass):
        raise SystemExit("Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in env.")

    # Ensure embedding model is set (defaults to HF MiniLM)
    os.environ.setdefault(
        "EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "huggingface")
    )

    singleton = GraphBuilderSingleton(neo4j_uri, neo4j_user, neo4j_pass, neo4j_db)

    def _load_chunks_from_json(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit("--chunks-json must contain a list of chunk objects")
        for item in data:
            if not isinstance(item, dict) or "text" not in item:
                raise SystemExit("Each chunk must be an object with at least a 'text' field")
        return data

    def _load_chunks_from_qdrant() -> List[Dict[str, Any]]:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue
        except Exception as e:
            raise SystemExit(
                f"qdrant-client is required for Qdrant mode: {e}. Install with: pip install qdrant-client"
            )
        if not args.qdrant_collection or not args.qdrant_url or not args.qdrant_doc_value:
            raise SystemExit(
                "For Qdrant mode, provide --qdrant-url, --qdrant-collection, --qdrant-doc-value"
            )
        client = QdrantClient(url=args.qdrant_url, api_key=(args.qdrant_api_key or None))
        flt = Filter(must=[FieldCondition(key=args.qdrant_doc_key, match=MatchValue(value=args.qdrant_doc_value))])
        out: List[Dict[str, Any]] = []
        next_page = None
        while True:
            points, next_page = client.scroll(
                collection_name=args.qdrant_collection,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=1000,
                offset=next_page,
            )
            for p in points:
                payload = p.payload or {}
                text = payload.get(args.qdrant_text_key)
                if not text:
                    continue
                item = {"text": text}
                if args.qdrant_order_key in payload:
                    item["position"] = payload.get(args.qdrant_order_key)
                if args.qdrant_page_key in payload:
                    item["page_number"] = payload.get(args.qdrant_page_key)
                out.append(item)
            if not next_page:
                break
        if not out:
            raise SystemExit("No chunks returned from Qdrant for the given filter")
        return out

    async def runner():
        external_mode = bool(args.chunks_json) or bool(args.qdrant_collection)
        if external_mode:
            # Determine file_name for document
            file_name = args.file_name.strip() or args.qdrant_doc_value.strip()
            if not file_name:
                raise SystemExit("Provide --file-name (or --qdrant-doc-value) when using external chunks")
            if args.chunks_json:
                chunks = _load_chunks_from_json(args.chunks_json)
            else:
                chunks = _load_chunks_from_qdrant()
            uri_latency, response = await singleton.run_with_external_chunks(
                model=args.model,
                file_name=file_name,
                chunks=chunks,
                allowed_nodes=args.allowed_nodes,
                allowed_relationship=args.allowed_relationship,
                additional_instructions=(args.additional_instructions or None),
            )
        else:
            if not args.s3_url:
                raise SystemExit("Provide --s3-url or use external chunk options like --chunks-json or Qdrant flags")
            uri_latency, response = await singleton.run_from_s3(
                s3_url=args.s3_url,
                aws_access_key_id=args.aws_access_key_id or None,
                aws_secret_access_key=args.aws_secret_access_key or None,
                b2_bucket=args.b2_bucket,
                model=args.model,
                allowed_nodes=args.allowed_nodes,
                allowed_relationship=args.allowed_relationship,
                token_chunk_size=args.token_chunk_size,
                chunk_overlap=args.chunk_overlap,
                chunks_to_combine=args.chunks_to_combine,
                backblaze_endpoint_url=args.b2_endpoint or None,
                additional_instructions=(args.additional_instructions or None),
            )
        print(json.dumps({"latency": uri_latency, "result": response}, indent=2))

    asyncio.run(runner())
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


