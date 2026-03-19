import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '50mb' }));

// In-memory vector store
let documentStore: { text: string; embedding: number[] }[] = [];

// API Routes
app.post("/api/ingest", async (req, res) => {
  try {
    const { chunks } = req.body; // Array of { text, embedding }
    if (!chunks || !Array.isArray(chunks)) {
      return res.status(400).json({ error: "Invalid chunks format" });
    }

    documentStore = [...documentStore, ...chunks];
    res.json({ success: true, count: chunks.length, total: documentStore.length });
  } catch (error) {
    console.error("Ingestion error:", error);
    res.status(500).json({ error: "Failed to ingest documents" });
  }
});

app.post("/api/query", async (req, res) => {
  try {
    const { queryEmbedding } = req.body;
    if (!queryEmbedding || !Array.isArray(queryEmbedding)) {
      return res.status(400).json({ error: "Query embedding is required" });
    }

    if (documentStore.length === 0) {
      return res.json({ context: [] });
    }

    // Calculate cosine similarity
    const similarities = documentStore.map(doc => {
      const dotProduct = doc.embedding.reduce((sum, val, i) => sum + val * queryEmbedding[i], 0);
      const mag1 = Math.sqrt(doc.embedding.reduce((sum, val) => sum + val * val, 0));
      const mag2 = Math.sqrt(queryEmbedding.reduce((sum, val) => sum + val * val, 0));
      return { text: doc.text, score: dotProduct / (mag1 * mag2) };
    });

    // Sort and take top K
    const topK = similarities
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    res.json({ 
      context: topK.map(t => t.text) 
    });
  } catch (error) {
    console.error("Query error:", error);
    res.status(500).json({ error: "Failed to process query" });
  }
});

app.post("/api/clear", (req, res) => {
  documentStore = [];
  res.json({ success: true });
});

async function startServer() {
  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
