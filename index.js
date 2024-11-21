import e from "express";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import dotenv from "dotenv";

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
  const query = req.body.query;

  const queryVector = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: query,
  });

  const queryResult = await pineconeClient.Index(indexName).query({
    vector: queryVector.data[0].embedding,
    topK: 5,
  });

  const context = queryResult.matches
    .map((match) => match.metadata.text)
    .join("\n");

  const prompt = `Prompt: ${query}\nCpntext: ${context}`;

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    prompt: prompt,
    max_tokens: 1024,
    n: 1,
    stop: null,
    temperature: 0.1,
  });

  res.json({ response: completion.data.choices[0].text });
});

app.get("/", (req, res) => {
  res.send("It works!");
});

app.listen(port, () => {
  console.log(`Server is running at port ${port}`);
});
