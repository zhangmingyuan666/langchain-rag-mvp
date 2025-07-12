import { initEmbedding } from "./embedding.js";

async function retriveByPrompt(vectorStore, prompt) {
    const retriever = vectorStore.asRetriever(3); // 设置返回最相似的 3 个文档
    const relatedDocs = await retriever.invoke(prompt); 
    return relatedDocs;
}

export {
    retriveByPrompt
}