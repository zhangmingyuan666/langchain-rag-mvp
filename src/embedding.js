import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
async function initEmbedding(chunks) {
    const embeddings = new OllamaEmbeddings({ model: "mxbai-embed-large" });
      
    // 根据外部 docsChunks (切割好的片段) 已经大模型自己的 embeddings 生成向量
    const vectorStore = await MemoryVectorStore.fromDocuments(
        chunks,           // 上一步拆分后得到的文档块数组
        embeddings        // Ollama 嵌入模型
    );

    return vectorStore;
}

export {
    initEmbedding
}
