require("dotenv").config();   // must run BEFORE ./llm is required

const express = require("express");
const { Pool } = require("pg");
const { callLLM } = require("./llm");
const { search } = require("./search");

// `pg` also reads PGHOST / PGUSER / PGPASSWORD from the environment if you set them.
const pool = new Pool({ database: "rag" });

/* ------------------------------------------------------------------
 * 1. THE TOOLS
 *
 * A "tool" is just a normal async function that talks to the database.
 * Nothing AI-specific about it. Rules that keep them usable later:
 *   - take plain arguments (a userId, a status), not a request object
 *   - return plain JSON-able data, or null
 *   - always use $1/$2 placeholders, never string concatenation
 * ------------------------------------------------------------------ */

async function getLatestSip(userId) {
  const result = await pool.query(
    `SELECT id, status, order_type, amount, frequency, created_at, next_debit_date
       FROM orders
      WHERE user_id = $1
        AND order_type = 'purchase_sip'
      ORDER BY created_at DESC
      LIMIT 1`,
    [userId]
  );
  return result.rows[0] ?? null;
}

// is_pause_active is computed by Postgres, not the model: models don't
// reliably know today's date and are bad at comparing dates.
async function getLatestPausedSip(userId) {
  const result = await pool.query(
    `SELECT id, status, order_type, amount, frequency, created_at,
            to_char(pause_till_date, 'YYYY-MM-DD') AS pause_till_date,
            coalesce(pause_till_date >= current_date, false) AS is_pause_active
       FROM orders
      WHERE user_id = $1
        AND order_type = 'pause_sip'
      ORDER BY created_at DESC
      LIMIT 1`,
    [userId]
  );
  return result.rows[0] ?? null;
}

// An optional argument: when `status` is null the filter is skipped.
// That is what `($2::text IS NULL OR status = $2)` does.
async function listOrders(userId, status = null) {
  const result = await pool.query(
    `SELECT id, status, order_type, amount, created_at
       FROM orders
      WHERE user_id = $1
        AND ($2::text IS NULL OR status = $2)
      ORDER BY created_at DESC
      LIMIT 20`,
    [userId, status]
  );
  return result.rows;
}

// An aggregate, so the model can answer "how much have I invested?"
async function getSipSummary(userId) {
  const result = await pool.query(
    `SELECT order_type,
            count(*)::int        AS order_count,
            coalesce(sum(amount), 0) AS total_amount
       FROM orders
      WHERE user_id = $1
        AND status = 'paid'
      GROUP BY order_type
      ORDER BY order_type`,
    [userId]
  );
  return result.rows;
}

/* ------------------------------------------------------------------
 * 2. THE REGISTRY
 *
 * `description` is what the model reads to decide WHICH tool fits the
 * question - write it for the model, like docs for a new colleague.
 * `parameters` is a JSON Schema of the arguments the model may send.
 *
 * Notice userId is NOT a parameter. The server passes it in (`run(args,
 * userId)`), so the model can never be talked into fetching another
 * user's data. The model picks the tool; your code picks whose data.
 * ------------------------------------------------------------------ */

const NO_ARGS = { type: "object", properties: {}, required: [] };

const tools = {
  getLatestSip: {
    description:
      "Get the user's most recent SIP (purchase_sip order): its status " +
      "(paid, failed, paused...), amount, frequency and next debit date. " +
      "Use for 'what is my SIP', 'is my SIP active/paused', 'next debit'.",
    parameters: NO_ARGS,
    run: (args, userId) => getLatestSip(userId),
  },
  getLatestPausedSip: {
    description:
      "Get the user's most recent request to pause a SIP (pause_sip order), " +
      "with pause_till_date and is_pause_active. is_pause_active is the " +
      "source of truth for whether the pause is still running today; " +
      "the status field alone does not say that.",
    parameters: NO_ARGS,
    run: (args, userId) => getLatestPausedSip(userId),
  },
  listOrders: {
    description:
      "List the user's recent orders of every type, newest first, " +
      "optionally filtered by status.",
    parameters: {
      type: "object",
      properties: {
        status: {
          type: "string",
          description: "Only return orders with this status, e.g. paid, failed, paused",
        },
      },
      required: [],
    },
    run: (args, userId) => listOrders(userId, args.status ?? null),
  },
  getSipSummary: {
    description:
      "Total amount and number of orders per order type, paid orders only. " +
      "Use for 'how much have I invested'.",
    parameters: NO_ARGS,
    run: (args, userId) => getSipSummary(userId),
  },
};

// The exact format the /chat/completions API expects in `tools`.
const toolSchemas = Object.entries(tools).map(([name, tool]) => ({
  type: "function",
  function: { name, description: tool.description, parameters: tool.parameters },
}));

/* ------------------------------------------------------------------
 * 3. THE SERVER
 * ------------------------------------------------------------------ */

const app = express();
app.use(express.json());

// Exactly what gets sent to the model as `tools`.
app.get("/tools", (req, res) => res.json(toolSchemas));

// Call a tool by hand, for testing: body is { userId, ...args }.
app.post("/tools/:name", async (req, res) => {
  const tool = tools[req.params.name];
  if (!tool) {
    return res.status(404).json({ error: `unknown tool: ${req.params.name}` });
  }

  try {
    const result = await tool.run(req.body ?? {}, req.body?.userId);
    res.json({ tool: req.params.name, result });
  } catch (err) {
    // Never let a raw DB error escape to the caller.
    console.error(`tool ${req.params.name} failed:`, err.message);
    res.status(500).json({ error: "tool failed", detail: err.message });
  }
});

/* ------------------------------------------------------------------
 * 4. CHAT WITH MEMORY + TOOLS
 *
 * The loop:
 *   1. send messages + tool list to the model
 *   2. model replies with EITHER text (done) OR "please call tool X"
 *   3. we run X, append its result as a { role: "tool" } message, go to 1
 *
 * Memory lives in a plain Map in RAM: restarting the server wipes it.
 * ------------------------------------------------------------------ */

// The rule covers EVERY question, not just ones with docs attached: when
// it only mentioned attached docs, a question with no matching docs
// ("what do you know about swp") was answered from general knowledge.
const SYSTEM_PROMPT =
  "You are a support assistant for a mutual-fund investing app. Be brief. " +
  "Answer ONLY from two sources: (1) the reference documents attached to " +
  "the user's message, and (2) the results of your tools, which look up " +
  "the user's own orders. Never answer from your own general knowledge, " +
  "even if you know the answer, and never describe app features that " +
  "these sources don't mention. If neither source covers the question, " +
  "say you don't have that information. Never guess or invent order " +
  "details; if a tool returns nothing, say so. Greetings and small talk " +
  "are fine.";

const MAX_HISTORY = 20;   // messages kept per conversation
const MAX_TOOL_STEPS = 5; // stop a model that keeps calling tools forever

/* ---- RAG: find policy docs related to the question ---- */

const RAG_LIMIT = 3;       // at most 3 chunks per question
// Weak matches are dropped. Measured on our docs: greetings and off-topic
// questions score 0.06-0.18, policy questions 0.56-0.79. Data questions
// ("what is my latest sip?") score ~0.5 because they share words with the
// docs, so they may get a harmless extra chunk - no cut-off splits those.
const RAG_MIN_SCORE = 0.4;

async function retrieveDocs(question) {
  try {
    return await search(question, { limit: RAG_LIMIT, minScore: RAG_MIN_SCORE });
  } catch (err) {
    // If Qdrant is down, still answer - just without docs.
    console.error("doc search failed:", err.message);
    return [];
  }
}

// The question the LLM sees: the docs first, clearly labelled as
// reference material, then the user's actual question.
function withDocs(question, docs) {
  // Say so explicitly when nothing matched - an empty search must not be
  // silent, or the model treats it as permission to use general knowledge.
  if (!docs.length) {
    return `Reference from our policy documents: none matched this question.\n\nQuestion: ${question}`;
  }
  const refs = docs.map((d, i) => `[${i + 1}] ${d.text.replace(/\n+/g, " ")}`).join("\n");
  return `Reference from our policy documents:\n${refs}\n\nQuestion: ${question}`;
}

const conversations = new Map(); // conversationId -> [{ role, content }, ...]

async function runToolCall(call, userId) {
  const tool = tools[call.function.name];
  if (!tool) return { error: `unknown tool ${call.function.name}` };

  try {
    // arguments arrive as a JSON *string*, and the model can write bad JSON
    const args = JSON.parse(call.function.arguments || "{}");
    return (await tool.run(args, userId)) ?? { result: "no matching orders" };
  } catch (err) {
    // Hand the error back to the model instead of crashing - it can
    // explain the problem or try something else.
    return { error: err.message };
  }
}

app.post("/chat", async (req, res) => {
  // In a real app userId comes from the login session, never the body.
  const { message, userId, conversationId = "default" } = req.body ?? {};
  if (!message || !userId) {
    return res
      .status(400)
      .json({ error: 'send {"userId": "...", "message": "...", "conversationId": "..."}' });
  }

  const history = conversations.get(conversationId) ?? [];
  const docs = await retrieveDocs(message);
  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    ...history,
    { role: "user", content: withDocs(message, docs) },
  ];
  const toolsUsed = [];

  try {
    let reply = null;

    for (let step = 0; step < MAX_TOOL_STEPS; step++) {
      const answer = await callLLM(messages, toolSchemas);


      // No tool requested: this is the final answer.
      if (!answer.tool_calls?.length) {
        reply = answer.content;
        break;
      }
      
      

      // The model's "please call these tools" turn must go into the list,
      // followed by one { role: "tool" } result per call, matched by id.
      messages.push({
        role: "assistant",
        content: answer.content ?? "",
        tool_calls: answer.tool_calls,
      });

      for (const call of answer.tool_calls) {
        const result = await runToolCall(call, userId);
        toolsUsed.push({ tool: call.function.name, args: call.function.arguments, result });
        messages.push({
          role: "tool",
          tool_call_id: call.id,
          content: JSON.stringify(result),
        });
      }
    }

    if (reply === null) reply = "Sorry, I couldn't finish looking that up.";

    // Memory keeps only the question and the final answer. The tool
    // messages are dropped: cheaper, and a follow-up just calls the tool
    // again, so it sees fresh data instead of a stale copy.
    // Save the plain question, not the version with docs pasted in -
    // otherwise every turn stores another copy of the docs.
    history.push({ role: "user", content: message });
    history.push({ role: "assistant", content: reply });
    conversations.set(conversationId, history.slice(-MAX_HISTORY));

    const sources = docs.map((d) => ({ source: d.source, section: d.section, score: d.score }));
    res.json({ reply, sources, toolsUsed, conversationId });
  } catch (err) {
    console.error("llm failed:", err.message);
    res.status(500).json({ error: "llm failed", detail: err.message });
  }
});

// See what the model "remembers" for a conversation.
app.get("/chat/:conversationId", (req, res) => {
  res.json(conversations.get(req.params.conversationId) ?? []);
});

// Forget a conversation.
app.delete("/chat/:conversationId", (req, res) => {
  conversations.delete(req.params.conversationId);
  res.json({ cleared: req.params.conversationId });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`listening on http://localhost:${PORT}`));
