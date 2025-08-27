import "dotenv/config";
import express from "express";
import cors from "cors";
import { z } from "zod";

// LangChain + Gemini
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createToolCallingAgent, AgentExecutor } from "langchain/agents";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// ---------- A) build a tiny knowledge base ----------

const rawNotes = `
GPUs (graphics processing units) excel at parallel computation.
They run many small threads together, great for linear algebra (matmul), graphics, and ML.
CUDA (NVIDIA) and ROCm (AMD) expose kernels, blocks, and warps/wavefronts for SIMT execution.
GPUs have high memory bandwidth; latency is hidden by massive concurrency.
For PhD-level: SIMT vs SIMD, memory hierarchy (global/shared/register), occupancy,
coalesced memory access, tensor cores, roofline performance modeling.
For kids: think of thousands of tiny helpers doing small jobs at the same time.
`;
// split into chunks + tag metadata (for citations)
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 400, chunkOverlap: 60 });
const docs = await splitter.createDocuments([rawNotes]);
docs.forEach((d, i) => {
  d.metadata = { source: "gpu_notes", chunk: i };
});
const embeddings = new GoogleGenerativeAIEmbeddings({ model: "text-embedding-004" });
const vectorstore = await MemoryVectorStore.fromDocuments(docs, embeddings);
const retriever=vectorstore.asRetriever({k:3})
// ---------- B) tools ----------
/** Retriever tool that returns JSON (so we can extract citations later) */

const kbTool=new DynamicStructuredTool({
 name:'gpu_kb',
 description:
    "Look up accurate facts about GPUs, CUDA/ROCm, memory hierarchy, etc.",
 schema:z.object({query:z.string()}),
 func:async({query})=>{
    const hits=await retriever.invoke(query)
    const results=hits.map(d=>({
        source:d.metadata?.source ?? 'unknown',
        chunk:d.metadata.chunk ?? null,
        text:d.pageContent.slice(0,300)
    }))
     // tools should return string or JSON-serializable
    return JSON.stringify(results);
 }
})
/** Simple calculator so the agent can choose between tools */
const CalcInput = z.object({
  a: z.number(),
  b: z.number(),
  op: z.enum(["add", "sub", "mul", "div", "pow"]),
});
const calculator = new DynamicStructuredTool({
  name: "calculator",
  description: "Do exact arithmetic (add, sub, mul, div, pow).",
  schema: CalcInput,
  func: async ({ a, b, op }) => {
    const val = { add: a + b, sub: a - b, mul: a * b, div: b === 0 ? "ERR" : a / b, pow: a ** b }[op];
    return String(val);
  },
});
// ---------- C) agent prompt (keep it simple) ----------
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a helpful expert.
- If the question needs facts about GPUs, use the "gpu_kb" tool.
- If the question needs math, use the "calculator" tool.
- Otherwise answer directly and be concise.
When you use tools, include the results in your final answer.`,
  ],
  ["human", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"), // required for tool-using agents
]);
// ---------- D) wire model + agent ----------
const llm = new ChatGoogleGenerativeAI({ model: "gemini-2.0-flash", temperature: 0.2 });
const tools = [kbTool, calculator];
const agent = await createToolCallingAgent({ llm, tools, prompt });
const executor = new AgentExecutor({
  agent,
  tools,
  verbose: true,
  returnIntermediateSteps: true, // so we can pull citations from tool outputs
});
const app = express();
app.use(cors());
app.use(express.json());

app.post("/ask", async (req, res) => {
  try {
        const input = String(req.body?.input ?? "");
    if (!input.trim()) return res.status(400).json({ error: "input is required" });
const citations = [];
const seen = new Set();
console.log(`object`)
const result = await executor.invoke({ input });
for (const step of result.intermediateSteps ?? []) {
  if (step.action?.tool === "gpu_kb" && typeof step.observation === "string") {
    try {
      const { results = [] } = JSON.parse(step.observation);
      for (const r of results) {
        const key = `${r.source}#${r.chunk}`;
        if (!seen.has(key)) {
          seen.add(key);
          citations.push({ key, source: r.source, chunk: r.chunk, snippet: r.text });
        }
      }
    } catch { /* log if you want */ }
  }
}
 res.json({
      answer: result.output,
      citations, // [{source:'gpu_notes', chunk:0, snippet:'...'}]
    });
       } catch {
          // ignore JSON parse errors
        }
      });

 const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`RAG agent API running on http://localhost:${PORT}`));
    
    