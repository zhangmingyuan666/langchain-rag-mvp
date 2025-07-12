import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";

async function createRetrivalQaChain(llm, vectorStore, query) {
    const prompt = ChatPromptTemplate.fromTemplate(
        `根据以下内容回答用户问题：{input}\n\n上下文：{context}`
    );
    
    const combineChain = await createStuffDocumentsChain({ llm, prompt });  // "填充"文档链
    
    // 从向量存储创建检索器
    const retriever = vectorStore.asRetriever();
    
    const retrievalChain = await createRetrievalChain({
        combineDocsChain: combineChain,
        retriever: retriever
    });
    
    const answer = await retrievalChain.invoke({ query });
    return answer;
}

export {
    createRetrivalQaChain
}