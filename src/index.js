import { OllamaEmbeddings } from "@langchain/ollama";
import { Ollama } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import {StringOutputParser} from '@langchain/core/output_parsers';
import {TextLoader} from 'langchain/document_loaders/fs/text';
import {CharacterTextSplitter} from '@langchain/textsplitters';

// 1. 嵌入模型
const embed_model = new OllamaEmbeddings({ model: "mxbai-embed-large" });

// 2. 构建向量数据库
const loader = new TextLoader("./src/sources/txt/situ-profile.txt");
const docs = await loader.load();
const splitter = new CharacterTextSplitter({        
  chunkSize: 500,       // 减小每块最大字符数
  chunkOverlap: 50,     // 减小块与块之间重叠字符数
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

// 6. 执行链
const result = await chain.invoke("who is situ yongcong");
console.log(result);
