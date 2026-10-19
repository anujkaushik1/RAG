const { QdrantClient } = require("@qdrant/js-client-rest");

// Local Qdrant, started from ./qdrant.
// 127.0.0.1, not "localhost": Node 18 tries IPv6 (::1) for localhost first,
// but Qdrant listens on IPv4 only, so "localhost" fails with ECONNREFUSED.
const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL || "http://127.0.0.1:6333",
});

// Default collection (like a table name) for our knowledge docs. Both
// ingest.js (writing) and search.js (reading) use it, so it lives here.
const COLLECTION = process.env.QDRANT_COLLECTION || "knowledge_mutual_funds";

module.exports = { qdrant, COLLECTION };
