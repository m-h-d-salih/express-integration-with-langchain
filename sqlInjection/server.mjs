import dotenv from 'dotenv';
import express from "express";
import cors from "cors";
import { z } from "zod";

import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createToolCallingAgent, AgentExecutor } from "langchain/agents";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
// LangChain + Google Gemini:
// ChatGoogleGenerativeAI: the chat model (e.g., gemini-2.0-flash).

// GoogleGenerativeAIEmbeddings: turns text → vectors for search.

// ChatPromptTemplate: builds chat prompts cleanly.

// MessagesPlaceholder: placeholder for tool-calling “scratchpad”.

// createToolCallingAgent: creates an agent that can call tools.

// AgentExecutor: runs the agent and returns results + tool steps.

// DynamicStructuredTool: define your own tools with validation.

// MemoryVectorStore: in-memory vector DB (resets on restart).

// RecursiveCharacterTextSplitter: splits text into chunks for retrieval.

import { pool } from "./db.mjs";
dotenv.config({path:'../.env'})
// ---------- A) vector plumbing ----------
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 400, chunkOverlap: 60 });
// Prepares a utility to break big text into ~400-char pieces with 60-char overlap.

// Overlap helps preserve context between chunks.

const embeddings = new GoogleGenerativeAIEmbeddings({ model: "text-embedding-004" });
// Sets up Gemini’s embedding model to convert text → numeric vectors.

// Needs process.env.GOOGLE_API_KEY.

let vectorstore;     // will be created from DB on startup
let retriever;       // vectorstore.asRetriever({ k: 3 })

async function buildIndexFromDB() {
  const [rows] = await pool.query("SELECT id, title, content FROM notes ORDER BY id");
  // Build texts + matching metadata arrays
  const texts = rows.map(r => `Title: ${r.title}\n\n${r.content}`);
  const metadatas = rows.map(r => ({ source: "mysql_notes", id: r.id, title: r.title }));

  const docs = await splitter.createDocuments(texts, metadatas);
//   Splits each text into smaller Documents and attaches matching metadata to each chunk.
  vectorstore = await MemoryVectorStore.fromDocuments(docs, embeddings);
//   Embeds all chunks and stores them in RAM.
  retriever = vectorstore.asRetriever({ k: 3 });
//   Creates a helper that, given a query, returns the top-3 similar chunks.

  console.log(`[RAG] Indexed ${rows.length} notes (chunks: ${docs.length}).`);
}

async function addNoteToIndex(note) {
  if (!vectorstore) await buildIndexFromDB(); // safety
  const docs = await splitter.createDocuments(
    [`Title: ${note.title}\n\n${note.content}`],
    [{ source: "mysql_notes", id: note.id, title: note.title }]
  );
//   Splits just this note into chunks with metadata.
  await vectorstore.addDocuments(docs);
  //   Adds the new chunks to the in-memory index .
  console.log(`[RAG] Added note #${note.id} to index (chunks: ${docs.length}).`);
}

// ---------- B) tools ----------
const kbTool = new DynamicStructuredTool({
  name: "gpu_kb",
  description:
    "Look up accurate facts about GPUs, CUDA/ROCm, memory hierarchy, etc. Input: a search query.",
  schema: z.object({ query: z.string() }),
  func: async ({ query }) => {
    const hits = await retriever.invoke(query); // modern retriever call
    // When the agent uses this tool, we run a semantic search over the vector store.
    const results = hits.map(d => ({
      source: d.metadata?.source ?? "unknown",
      id: d.metadata?.id ?? null,
      title: d.metadata?.title ?? null,
      text: d.pageContent.slice(0, 300),
    }));
    // IMPORTANT: return an object wrapper so downstream parsing sees `results`
    return JSON.stringify({ results });
  },
});

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

// ---------- C) agent prompt ----------
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a helpful expert.
- If the question needs facts about GPUs, use "gpu_kb".
- If the question needs math, use "calculator".
- Otherwise answer directly and be concise.
When you use tools, integrate the results into the final answer.`,
  ],
  ["human", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);
// System message: tells the model when to use which tool and to keep answers concise.

// Human message: the user’s input will fill {input} at runtime.

// agent_scratchpad: a required placeholder where the agent writes tool calls/results during reasoning.

// ---------- D) model + agent ----------
const llm = new ChatGoogleGenerativeAI({ model: "gemini-2.0-flash", temperature: 0.2 });
const tools = [kbTool, calculator];
// List of tools the agent is allowed to call.
const agent = await createToolCallingAgent({ llm, tools, prompt });
// Builds a tool-using agent that can decide when/how to call tools.
const executor = new AgentExecutor({
  agent,
  tools,
  verbose: true,
  returnIntermediateSteps: true,
});
// Wraps the agent so you can call executor.invoke({ input }).

// verbose: true: logs decisions to console.

// returnIntermediateSteps: true: returns tool call traces so you can extract citations.

// build initial index from DB
await buildIndexFromDB();
// Build the search index once at startup from MySQL.

// ---------- E) express API ----------
const app = express();
app.use(cors());
app.use(express.json());

app.post("/ask", async (req, res) => {
  try {
    const input = String(req.body?.input ?? "");
    if (!input.trim()) return res.status(400).json({ error: "input is required" });

    const result = await executor.invoke({ input });
// Runs the agent with your input.

// result contains the final answer and (because of returnIntermediateSteps) the tool traces.

    // extract citations from gpu_kb tool calls
    const citations = [];
    const seen = new Set();
    for (const step of result.intermediateSteps ?? []) {
      if (step.action?.tool === "gpu_kb" && typeof step.observation === "string") {
        try {
          const { results = [] } = JSON.parse(step.observation);
          for (const r of results) {
            const key = `${r.source}#${r.id}`;
            if (!seen.has(key)) {
              seen.add(key);
              citations.push({ key, source: r.source, id: r.id, title: r.title, snippet: r.text });
            }
          }
        } catch {}
      }
    }

    res.json({ answer: result.output, citations });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e?.message || "unknown error" });
  }
});
// Create a new note and index it immediately
app.post("/notes", async (req, res) => {
  try {
    const { title, content } = req.body || {};
    if (!title || !content) return res.status(400).json({ error: "title and content are required" });

    const [r] = await pool.query(
      "INSERT INTO notes (title, content) VALUES (?, ?)",
      [title, content]
    );
    const note = { id: r.insertId, title, content };
    await addNoteToIndex(note);

    res.status(201).json({ id: r.insertId });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e?.message || "unknown error" });
  }
});
// List notes (for quick sanity check)
app.get("/notes", async (_req, res) => {
  const [rows] = await pool.query("SELECT id, title, LEFT(content, 120) AS preview FROM notes ORDER BY id DESC");
  res.json(rows);
});
// Rebuild the whole vector index from DB (optional admin)
app.post("/reindex", async (_req, res) => {
  await buildIndexFromDB();
  res.json({ ok: true });
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`RAG agent API running on http://localhost:${PORT}`));