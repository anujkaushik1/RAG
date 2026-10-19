/* ------------------------------------------------------------------
 * THE LLM CLIENT
 *
 * Every provider below speaks the same "/chat/completions" shape, so
 * swapping provider is 3 env vars, not a code change:
 *
 *   Groq (free)        base https://api.groq.com/openai/v1
 *                      model openai/gpt-oss-120b   key from console.groq.com/keys
 *   Google             base https://generativelanguage.googleapis.com/v1beta/openai
 *                      model gemini-3.8-flash      (needs billing credits, not free)
 *   OpenRouter (free)  base https://openrouter.ai/api/v1
 *   DeepSeek (paid)    base https://api.deepseek.com/v1       model deepseek-chat
 *   Ollama (local)     base http://localhost:11434/v1         key can be anything
 *
 * `messages` is a list of turns: { role, content }, where role is
 * "system" (instructions), "user" (them) or "assistant" (the model).
 * The API is stateless - you resend the whole list every time.
 * ------------------------------------------------------------------ */

const LLM = {
  baseUrl: process.env.LLM_BASE_URL || "https://api.groq.com/openai/v1",
  apiKey: process.env.LLM_API_KEY || "",
  model: process.env.LLM_MODEL || "openai/gpt-oss-120b",
};

// Loud warning at startup - otherwise a missing key only shows up later
// as a confusing 401 from the provider.
if (!LLM.apiKey) {
  console.warn("WARNING: LLM_API_KEY is not set - /chat will fail with 401");
} else {
  console.log(`llm: ${LLM.model} via ${LLM.baseUrl}`);
}

// Returns the model's whole message, not just its text, because when the
// model wants a tool the answer is in `message.tool_calls`, not `content`.
async function callLLM(messages, tools) {
  const body = { model: LLM.model, messages };
  if (tools?.length) body.tools = tools;
  const response = await fetch(`${LLM.baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${LLM.apiKey}`,
    },
    body: JSON.stringify(body),
  });

  // fetch does NOT throw on 401/429/500 - you have to check this yourself.
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`LLM ${response.status}: ${text.slice(0, 300)}`);
  }

  const data = await response.json();
  return data.choices[0].message;
}

module.exports = { callLLM, LLM };
