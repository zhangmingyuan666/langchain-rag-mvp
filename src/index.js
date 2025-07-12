/*
 * @Author: zhangmingyuan 2369558390@qq.com
 * @Date: 2025-07-12 14:04:32
 * @LastEditors: zhangmingyuan 2369558390@qq.com
 * @LastEditTime: 2025-07-12 14:04:34
 * @FilePath: /testing-langchainjs/src/index.js
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import { OllamaEmbeddings } from "@langchain/ollama";
import { Ollama } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from '@langchain/core/output_parsers';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CharacterTextSplitter } from '@langchain/textsplitters';
import readline from 'readline';

// 1. 嵌入模型
const embed_model = new OllamaEmbeddings({ model: "mxbai-embed-large" });

// 2. 构建向量数据库
const loader = new TextLoader("./src/sources/txt/situ-profile.txt");
const docs = await loader.load();

const splitter = new CharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const chunks = await splitter.splitDocuments(docs);

const vector_store = await MemoryVectorStore.fromTexts(
  chunks.map(chunk => chunk.pageContent),
  chunks.map(chunk => chunk.metadata), // metadata
  embed_model
);

// 3. LLM
const llm = new Ollama({
  model: "llama3",
  baseUrl: "http://127.0.0.1:11434"
});

// 4. Prompt
const SOME_PROMPT = PromptTemplate.fromTemplate(`
Answer the following question: {question}
using the given context:
{docs}
please answer in Chinese, an concat a "\n --- 喵" in the end of the answer.
`);

// 5. LCEL Chain
const chain = RunnableSequence.from([
  {
    docs: vector_store.asRetriever({
      k: 3, // 只返回相似度最高的前3个文档
      searchType: "similarity" // 使用相似度搜索
    }),
    question: new RunnablePassthrough()
  },
  SOME_PROMPT,
  llm,
  new StringOutputParser()
]);

// 6. 创建命令行接口
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

console.log("🤖 LangChain.js + Ollama Q&A System");
console.log("💡 Type your question and press Enter (type 'quit' to exit)");
console.log("📚 Available context: Situ Yongcong's profile");
console.log("=" .repeat(50));

// 7. 交互式问答循环
async function askQuestion() {
  rl.question('\n❓ Your question: ', async (input) => {
    if (input.toLowerCase() === 'quit' || input.toLowerCase() === 'exit') {
      console.log('\n👋 Goodbye!');
      rl.close();
      return;
    }

    if (input.trim() === '') {
      console.log('⚠️  Please enter a question.');
      askQuestion();
      return;
    }

    try {
      console.log('\n🤔 Thinking...');
      const result = await chain.invoke(input);
      console.log('\n🤖 Answer:');
      console.log(result);
    } catch (error) {
      console.error('\n❌ Error:', error.message);
    }

    // 继续下一轮问答
    askQuestion();
  });
}

// 8. 启动交互式问答
askQuestion();
