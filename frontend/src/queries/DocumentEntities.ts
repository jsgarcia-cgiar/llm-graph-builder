// Cypher query to visualize only entity nodes and their relationships for a single document
// Param: $document_name (string) - exact fileName of the document
// Returns: nodes (only entities) and rels (only relationships where both ends are entities)

export const DOCUMENT_ENTITIES_QUERY = `
MATCH docs = (d:Document {status:'Completed'})
WHERE d.fileName = $document_name
WITH docs, d ORDER BY d.createdAt DESC
CALL { WITH d
  OPTIONAL MATCH chunks=(d)<-[:PART_OF|FIRST_CHUNK]-(c:Chunk)
  RETURN chunks, c LIMIT 50
}
WITH []
// Collect only entity nodes and relationships among entities
+ collect {
  OPTIONAL MATCH (c:Chunk)-[:HAS_ENTITY]->(e:__Entity__), p=(e)-[*0..1]-(:__Entity__)
  RETURN p
}
AS paths
CALL { WITH paths UNWIND paths AS path UNWIND nodes(path) as node RETURN collect(distinct node) as nodes }
CALL { WITH paths UNWIND paths AS path UNWIND relationships(path) as rel RETURN collect(distinct rel) as rels }
// Ensure we only return entity nodes and relationships between entities
WITH [n IN nodes WHERE n:__Entity__] AS nodes, [r IN rels WHERE startNode(r):__Entity__ AND endNode(r):__Entity__] AS rels
RETURN nodes, rels`;

export default DOCUMENT_ENTITIES_QUERY;


