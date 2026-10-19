/* ------------------------------------------------------------------
 * CHUNKING: knowledge/*.md  ->  small pieces of text
 *
 * Uses LangChain's RecursiveCharacterTextSplitter in markdown mode.
 * It cuts by SIZE: it tries to cut at headings first, then blank
 * lines, then line breaks, then spaces - whatever keeps each piece
 * under chunkSize characters. Small neighbouring pieces get merged
 * back together up to chunkSize.
 * ------------------------------------------------------------------ */

const fs = require("fs");
const path = require("path");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");

const KNOWLEDGE_DIR = path.join(__dirname, "knowledge");

// 150: small enough that every short section of our docs stays in its
// own chunk (at 250, sections like Cutoff + Duration get glued together).
// 80 overlap: the splitter cuts at line breaks and our lines are ~75
// characters, so overlap must exceed one line or it does nothing.
const splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
  chunkSize: 150,
  chunkOverlap: 80,
});

async function chunkMarkdown(markdown, source) {
  const title = markdown.match(/^# (.+)$/m)?.[1].trim() ?? source;

  // Blank out the "# Title" line (keeping the line itself, so line
  // numbers still match the file) - we add the title back below.
  const body = markdown.replace(/^# .+$/m, "");

  // For every line, remember which "## " section it belongs to.
  const lines = body.split("\n");
  const sectionOfLine = [];
  let current = "Overview";
  for (const line of lines) {
    if (line.startsWith("## ")) current = line.slice(3).trim();
    sectionOfLine.push(current);
  }

  const docs = await splitter.createDocuments([body]);

  return docs.map((doc, i) => {
    const { from, to } = doc.metadata.loc.lines; // 1-based line numbers
    const section = sectionOfLine[from - 1];
    // The splitter knows nothing about our headings, so a chunk cut from
    // the middle of a section would lose its label. Put it back up front.
    const content = doc.pageContent.replace(/^## .+\n?/, "").trim();
    return {
      id: `${source}#${i}`,
      source,
      title,
      section,
      lines: `${from}-${to}`,
      text: `${title} > ${section}\n\n${content}`, // what gets embedded
    };
  });
}

async function loadChunks(dir = KNOWLEDGE_DIR) {
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".md")).sort();
  const perFile = await Promise.all(
    files.map((f) => chunkMarkdown(fs.readFileSync(path.join(dir, f), "utf8"), f))
  );
  return perFile.flat();
}

module.exports = { loadChunks, chunkMarkdown };

// `node chunk.js` prints the chunks so you can look at them.
if (require.main === module) {
  loadChunks().then((chunks) => {
    for (const c of chunks) {
      console.log(`----- ${c.id}  lines ${c.lines}  (${c.text.length} chars)`);
      console.log(c.text, "\n");
    }
    console.log(`${chunks.length} chunks from ${new Set(chunks.map((c) => c.source)).size} files`);
  });
}
