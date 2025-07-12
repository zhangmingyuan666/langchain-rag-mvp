/*
 * @Author: zhangmingyuan 2369558390@qq.com
 * @Date: 2025-07-12 14:04:32
 * @LastEditors: zhangmingyuan 2369558390@qq.com
 * @LastEditTime: 2025-07-12 14:04:34
 * @FilePath: /testing-langchainjs/src/index.js
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import { Ollama } from "@langchain/ollama";
import {initEmbedding} from './embedding.js';
import {splitTextToChunk} from './text-split.js';
import {createRetrivalQaChain} from './retrival-qa-chain.js';

export const query = "司徒永聪是谁"

async function main() {
  // 1. 加载和拆分文档
  const textChunks = await splitTextToChunk();

  console.log('chunk 切割成功', textChunks)

  // 2. 把 chunk 转化为向量
  const vectorStore = await initEmbedding(textChunks);
  console.log('向量转化成功', vectorStore)

  // 3. 初始化大模型，并设置检索链
  const llm = new Ollama({
    model: "llama3",      // 使用已安装的模型名称
    temperature: 0,
    baseUrl: "http://127.0.0.1:11434", // 使用 IPv4 地址
  });
  console.log('大模型初始化成功', llm)
  // 4. 执行检索链
  const answer = await createRetrivalQaChain(llm, vectorStore, query);
  console.log(answer);
}

main()
