from typing import Any
import types

__all__ = [
    "processing_source",
    "create_graph_database_connection",
    "get_documents_from_s3",
    "graphDBdataAccess",
    "sourceNode",
]


async def processing_source(
    uri: str,
    userName: str,
    password: str,
    database: str,
    model: str,
    file_name: str,
    pages: Any,
    allowedNodes: str,
    allowedRelationship: str,
    token_chunk_size: int,
    chunk_overlap: int,
    chunks_to_combine: int,
    is_uploaded_from_local: Any = None,
    merged_file_path: Any = None,
    retry_condition: Any = None,
    additional_instructions: Any = None,
):
    from src.main import processing_source as _processing_source
    return await _processing_source(
        uri,
        userName,
        password,
        database,
        model,
        file_name,
        pages,
        allowedNodes,
        allowedRelationship,
        token_chunk_size,
        chunk_overlap,
        chunks_to_combine,
        is_uploaded_from_local,
        merged_file_path,
        retry_condition,
        additional_instructions,
    )


def create_graph_database_connection(uri: str, userName: str, password: str, database: str):
    from src.shared.common_fn import create_graph_database_connection as _conn
    return _conn(uri, userName, password, database)


def get_documents_from_s3(
    s3_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    endpoint_url: str | None = None,
    bucket: str | None = None,
):
    from src.document_sources.s3_bucket import get_documents_from_s3 as _get
    return _get(
        s3_url,
        aws_access_key_id,
        aws_secret_access_key,
        endpoint_url=endpoint_url,
        bucket=bucket,
    )


def __getattr__(name: str):
    if name == "graphDBdataAccess":
        from src.graphDB_dataAccess import graphDBdataAccess as _C
        return _C
    if name == "sourceNode":
        from src.entities.source_node import sourceNode as _C
        return _C
    raise AttributeError(name)


