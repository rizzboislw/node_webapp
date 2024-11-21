import e from "express";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import dotenv from "dotenv";
import { assistantInstructions } from "./instruction.js";

dotenv.config();

const app = e();
const port = 3000;

app.use(e.json());

const pineconeClient = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const indexName = "beverage-data";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.post("/query", async (req, res) => {
  try {
    const query = req.body.query;

    if (!query || typeof query !== "string") {
      return res
        .status(400)
        .json({ error: "Query is required and must be a string." });
    }

    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: query,
    });

    const queryVector = embeddingResponse.data[0].embedding;

    const queryResult = await pineconeClient.Index(indexName).query({
      vector: queryVector,
      topK: 5,
      includeMetadata: true,
    });

    if (!queryResult.matches || queryResult.matches.length === 0) {
      console.warn("No matches found in Pinecone.");
      return res
        .status(404)
        .json({ response: "No relevant context found for your query." });
    }

    const context = queryResult.matches
      .filter((match) => match.metadata && match.metadata.text)
      .map((match) => match.metadata.text)
      .join("\n");

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `${assistantInstructions}\nContext:\n${context}`,
        },
        { role: "user", content: query },
      ],
      max_tokens: 1024,
      temperature: 0.4,
    });

    const responseText = completion.choices[0].message.content;
    res.json({ response: responseText });
  } catch (error) {
    console.error("Error in /query:", error.message);
    res.status(500).json({ error: error.message });
  }
});

app.get("/", (req, res) => {
  res.send("It works!");
});

app.listen(port, () => {
  console.log(`Server is running at port ${port}`);
});
