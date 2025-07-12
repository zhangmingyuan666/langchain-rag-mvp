import { OllamaEmbeddings } from "@langchain/ollama";
import { Ollama } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from '@langchain/core/output_parsers';
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// 准备数据
const texts = ["司徒永聪是一名软件开发者，专注于系统运维和前端开发。"];
const embed_model = new OllamaEmbeddings({ model: "mxbai-embed-large" });
const vector_store = await MemoryVectorStore.fromTexts(texts, [{}], embed_model);
const llm = new Ollama({ model: "llama3", baseUrl: "http://127.0.0.1:11434" });

console.log("=== 传统方式 vs LCEL 方式对比 ===\n");

// 方式1：传统链式调用
console.log("1. 传统方式（命令式）：");
console.log("   - 需要手动创建每个组件");
console.log("   - 需要手动连接组件");
console.log("   - 错误处理复杂");
console.log("   - 代码分散，难以理解整体流程\n");

// 方式2：LCEL 方式
console.log("2. LCEL 方式（声明式）：");
console.log("   - 直接声明整个流程");
console.log("   - 自动处理数据流");
console.log("   - 统一的错误处理");
console.log("   - 代码集中，流程清晰\n");

// 实际代码对比
console.log("=== 代码对比 ===\n");

console.log("传统方式代码：");
console.log(`
const prompt = ChatPromptTemplate.fromTemplate("回答：{input}\\n上下文：{context}");
const combineChain = await createStuffDocumentsChain({ llm, prompt });
const retrievalChain = await createRetrievalChain({ 
  combineDocsChain: combineChain, 
  retriever: vector_store.asRetriever() 
});
const answer = await retrievalChain.invoke({ query: "问题" });
`);

console.log("\nLCEL 方式代码：");
console.log(`
const chain = RunnableSequence.from([
  {
    docs: vector_store.asRetriever(),
    question: new RunnablePassthrough()
  },
  PromptTemplate.fromTemplate("回答：{question}\\n上下文：{docs}"),
  llm,
  new StringOutputParser()
]);
const result = await chain.invoke("问题");
`);

console.log("\n=== 主要优势 ===\n");
console.log("LCEL 的优势：");
console.log("1. 更简洁：减少了中间变量和步骤");
console.log("2. 更直观：整个流程像管道一样清晰");
console.log("3. 更灵活：容易添加、删除或重新排序步骤");
console.log("4. 更现代：这是 LangChain 推荐的未来发展方向");
console.log("5. 更好的调试：统一的错误处理和日志"); 