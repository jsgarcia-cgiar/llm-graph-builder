import os
import json
import argparse
import logging
import asyncio
from typing import Optional, Tuple, Dict, Any

# Reuse internal pipeline without modifying it
from src.main import processing_source
from src.document_sources.s3_bucket import get_documents_from_s3
from src.shared.common_fn import create_graph_database_connection
from src.graphDB_dataAccess import graphDBdataAccess
from src.entities.source_node import sourceNode
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


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Run LLM Graph Builder pipeline on a single S3/Backblaze file"
    )
    parser.add_argument(
        "--s3-url",
        required=True,
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

    async def runner():
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
