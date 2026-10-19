/* ------------------------------------------------------------------
 * EMBEDDINGS: text -> vector (a list of numbers)
 *
 * The vector captures the MEANING of the text, so two texts that mean
 * similar things get vectors that point in a similar direction - even
 * if they share no words ("pause my SIP" vs "stop my monthly investment").
 *
 * The model runs on this laptop - no API key, no rate limit. It is
 * downloaded once (~23 MB) on first use, then cached.
 *
 * all-MiniLM-L6-v2: small, fast, English. Every vector has 384 numbers.
 * ------------------------------------------------------------------ */

const { pipeline } = require("@huggingface/transformers");

const MODEL = "Xenova/all-MiniLM-L6-v2";
const DIMENSIONS = 384; // Qdrant needs this number when creating a collection

// Loading the model takes a second or two, so load it once and reuse it.
let extractorPromise = null;
function getExtractor() {
  extractorPromise ??= pipeline("feature-extraction", MODEL, { dtype: "q8" });
  return extractorPromise;
}

async function embed(text) {
  const extractor = await getExtractor();
  // pooling "mean": the model makes one vector per word; average them
  //                 into a single vector for the whole text.
  // normalize: scale every vector to length 1, so comparing two vectors
  //            only measures direction (meaning), not text length.
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data); // plain JS array of 384 numbers
}

module.exports = { embed, DIMENSIONS, MODEL };

// `node embed.js "some text"` prints its vector.
if (require.main === module) {
  const text = process.argv[2] || "How do I pause my SIP?";
  embed(text).then((vector) => {
    console.log(`text:   "${text}"`);
    console.log(`length: ${vector.length} numbers`);
    console.log(`first 8: [${vector.slice(0, 8).map((n) => n.toFixed(4)).join(", ")}, ...]`);
  });
}
