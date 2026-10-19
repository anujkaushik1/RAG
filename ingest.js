/* ------------------------------------------------------------------
 * INGEST: chunk -> embedding -> stored in Qdrant
 *
 * Each chunk becomes one Qdrant "point":
 *   id      - a unique id for the point
 *   vector  - the 384 numbers from embed(), used for searching
 *   payload - the chunk's text and labels, stored alongside so a
 *             search result can hand the text straight to the LLM
 * ------------------------------------------------------------------ */

const crypto = require("crypto");
const { qdrant, COLLECTION } = require("./qdrant");
const { embed, DIMENSIONS } = require("./embed");
const { loadChunks } = require("./chunk");

// Qdrant only accepts a number or a UUID as a point id, but our chunk
// ids look like "sip-pause-policy.md#2". So we turn each chunk id into a
// UUID - always the SAME UUID for the same chunk id. That way ingesting
// a chunk twice overwrites it instead of storing a duplicate.
function pointId(chunkId) {
  const h = crypto.createHash("md5").update(chunkId).digest("hex");
  return [h.slice(0, 8), h.slice(8, 12), h.slice(12, 16), h.slice(16, 20), h.slice(20, 32)].join("-");
}

// A collection is like a table. Create it on first use. `size` must match
// the embedding model (384), and "Cosine" compares vectors by direction,
// i.e. by meaning.
async function ensureCollection(collection = COLLECTION) {
  const { exists } = await qdrant.collectionExists(collection);
  if (!exists) {
    await qdrant.createCollection(collection, {
      vectors: { size: DIMENSIONS, distance: "Cosine" },
    });
    console.log(`created collection "${collection}"`);
  }
}

// Store many chunks: embed each one, then save them all in one request.
async function ingestChunks(chunks, collection = COLLECTION) {
  await ensureCollection(collection);

  const points = [];
  for (const chunk of chunks) {
    points.push({
      id: pointId(chunk.id),
      vector: await embed(chunk.text),
      payload: {
        chunk_id: chunk.id,
        source: chunk.source,
        title: chunk.title,
        section: chunk.section,
        lines: chunk.lines,
        text: chunk.text,
      },
    });
  }

  // wait: true -> only return once Qdrant has actually saved them
  await qdrant.upsert(collection, { wait: true, points });
  return points.length;
}

// Store one chunk.
async function ingestChunk(chunk, collection = COLLECTION) {
  return ingestChunks([chunk], collection);
}

// Rebuild the whole collection from knowledge/*.md. It deletes the old
// collection first: if a doc got shorter, its old extra chunks would
// otherwise stay in Qdrant and keep turning up in searches.
async function ingestKnowledge(collection = COLLECTION) {
  const { exists } = await qdrant.collectionExists(collection);
  if (exists) await qdrant.deleteCollection(collection);

  const chunks = await loadChunks();
  return ingestChunks(chunks, collection);
}

module.exports = { ingestChunk, ingestChunks, ingestKnowledge, ensureCollection };

// `node ingest.js` (re)loads every doc in knowledge/ into Qdrant.
// `node ingest.js my_collection` uses a different collection name.
if (require.main === module) {
  const collection = process.argv[2] || COLLECTION;
  const started = Date.now();
  ingestKnowledge(collection)
    .then((n) => console.log(`stored ${n} chunks in "${collection}" (${Date.now() - started} ms)`))
    .catch((err) => {
      console.error("ingest failed:", err.message);
      process.exit(1);
    });
}
