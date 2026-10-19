/* ------------------------------------------------------------------
 * SEARCH: question -> the most similar chunks in Qdrant
 *
 * The question is embedded with the SAME model as the chunks, then
 * Qdrant returns the chunks whose vectors point in the most similar
 * direction. score: 1 = same meaning, 0 = unrelated.
 * ------------------------------------------------------------------ */

const { qdrant, COLLECTION } = require("./qdrant");
const { embed } = require("./embed");

// Find the `limit` chunks closest in meaning to `question`.
//   minScore - drop weak matches (e.g. 0.3), so an off-topic question
//              returns nothing instead of the "least bad" chunks
//   source   - only search one file, e.g. "sip-pause-policy.md"
async function search(question, { limit = 3, minScore, source, collection = COLLECTION } = {}) {
  const vector = await embed(question);

  const { points } = await qdrant.query(collection, {
    query: vector,
    limit,
    with_payload: true, // give back the stored text and labels too
    score_threshold: minScore,
    filter: source ? { must: [{ key: "source", match: { value: source } }] } : undefined,
  });

  return points.map((p) => ({
    score: Number(p.score.toFixed(3)),
    source: p.payload.source,
    section: p.payload.section,
    lines: p.payload.lines,
    text: p.payload.text,
  }));
}

// How many chunks are stored.
async function countChunks(collection = COLLECTION) {
  const { count } = await qdrant.count(collection, { exact: true });
  return count;
}

module.exports = { search, countChunks };

// `node search.js "how many times can I pause"` prints the top matches.
if (require.main === module) {
  const question = process.argv[2] || "How long can I pause my SIP?";
  search(question, { limit: 3 })
    .then((results) => {
      console.log(`question: "${question}"\n`);
      for (const r of results) {
        console.log(`${r.score}  ${r.source} > ${r.section}  (lines ${r.lines})`);
        console.log(`       ${r.text.split("\n").slice(2).join(" ").trim()}\n`);
      }
    })
    .catch((err) => {
      console.error("search failed:", err.message);
      process.exit(1);
    });
}
