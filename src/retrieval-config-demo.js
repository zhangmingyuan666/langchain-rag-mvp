import { OllamaEmbeddings } from "@langchain/ollama";
import { Ollama } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from '@langchain/core/output_parsers';

// 准备测试数据
const texts = [
  "司徒永聪是一名软件开发者，专注于系统运维和前端开发。",
  "他经营个人博客 situ2001.com/blog，分享技术实践。",
  "他擅长 QNAP NAS 调优和自动化脚本开发。",
  "他使用 GTD、Obsidian、Logseq 等工具提升生产力。",
  "他喜欢旅行，记录生活体验和技术成长。"
];

const embed_model = new OllamaEmbeddings({ model: "mxbai-embed-large" });
const vector_store = await MemoryVectorStore.fromTexts(texts, [{}], embed_model);
const llm = new Ollama({ model: "llama3", baseUrl: "http://127.0.0.1:11434" });

console.log("=== 检索配置对比 ===\n");

// 1. 默认检索（不指定数量）
console.log("1. 默认检索（返回所有相关文档）：");
const defaultChain = RunnableSequence.from([
  {
    docs: vector_store.asRetriever(), // 默认设置
    question: new RunnablePassthrough()
  },
  PromptTemplate.fromTemplate("基于以下文档回答问题：{question}\n\n文档：{docs}"),
  llm,
  new StringOutputParser()
]);

// 2. 指定检索数量
console.log("2. 指定检索前3个文档：");
const top3Chain = RunnableSequence.from([
  {
    docs: vector_store.asRetriever({
      k: 3, // 只返回相似度最高的前3个文档
      searchType: "similarity" // 使用相似度搜索
    }),
    question: new RunnablePassthrough()
  },
  PromptTemplate.fromTemplate("基于以下文档回答问题：{question}\n\n文档：{docs}"),
  llm,
  new StringOutputParser()
]);

// 3. 高级检索配置
console.log("3. 高级检索配置：");
const advancedChain = RunnableSequence.from([
  {
    docs: vector_store.asRetriever({
      k: 2, // 返回前2个
      searchType: "mmr", // 使用最大边际相关性搜索
      fetchK: 4, // 先获取4个，然后选择最相关的2个
      lambda: 0.5 // MMR 多样性参数
    }),
    question: new RunnablePassthrough()
  },
  PromptTemplate.fromTemplate("基于以下文档回答问题：{question}\n\n文档：{docs}"),
  llm,
  new StringOutputParser()
]);

// 4. 自定义检索器
console.log("4. 自定义检索器（类似传统方式）：");
const customRetriever = vector_store.asRetriever({
  k: 3,
  searchType: "similarity",
  filter: null, // 可以添加过滤条件
  scoreThreshold: 0.7 // 相似度阈值
});

const customChain = RunnableSequence.from([
  {
    docs: customRetriever,
    question: new RunnablePassthrough()
  },
  PromptTemplate.fromTemplate("基于以下文档回答问题：{question}\n\n文档：{docs}"),
  llm,
  new StringOutputParser()
]);

console.log("=== 检索配置选项说明 ===\n");
console.log("可用的检索配置参数：");
console.log("- k: 返回的文档数量");
console.log("- searchType: 'similarity' | 'mmr' (最大边际相关性)");
console.log("- fetchK: MMR 搜索时先获取的文档数量");
console.log("- lambda: MMR 多样性参数 (0-1)");
console.log("- scoreThreshold: 相似度阈值");
console.log("- filter: 文档过滤条件");

console.log("\n=== 使用建议 ===\n");
console.log("1. 简单场景：使用默认检索");
console.log("2. 性能优化：指定 k 值限制返回数量");
console.log("3. 质量提升：使用 MMR 搜索获得更相关的文档");
console.log("4. 精确控制：设置相似度阈值过滤低质量匹配");

// 测试不同配置的效果
const testQuestion = "司徒永聪的技术专长是什么？";

console.log(`\n测试问题：${testQuestion}\n`);

try {
  console.log("默认检索结果：");
  const defaultResult = await defaultChain.invoke(testQuestion);
  console.log(defaultResult.substring(0, 200) + "...\n");
  
  console.log("前3个文档检索结果：");
  const top3Result = await top3Chain.invoke(testQuestion);
  console.log(top3Result.substring(0, 200) + "...\n");
  
} catch (error) {
  console.log("测试时出现错误：", error.message);
} 