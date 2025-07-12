# LangChain.js + Ollama Intelligent Q&A System

A local intelligent question-answering system based on LangChain.js and Ollama, supporting document retrieval and natural language Q&A.

## 🚀 Features

-   **Local Deployment**: Uses Ollama local large models, no internet required
-   **Document Retrieval**: Supports text chunking and vectorized retrieval
-   **Modern Architecture**: Built with LangChain Expression Language (LCEL)
-   **Chinese Support**: Fully supports Chinese Q&A
-   **Flexible Configuration**: Customizable retrieval parameters and model settings
-   **Interactive CLI**: Command-line interface with conversation history and commands
-   **Enhanced Context Awareness**: Advanced conversation continuity and context management

## 📋 System Requirements

-   Node.js 18+
-   Ollama installed and running
-   At least 4GB available memory (for running large models)

## 🛠️ Installation Steps

### 1. Clone the project

```bash
git clone <your-repo-url>
cd testing-langchainjs
```

### 2. Install dependencies

```bash
npm install
```

### 3. Install Ollama models

```bash
# Install large language model
ollama pull llama3

# Install embedding model
ollama pull mxbai-embed-large
```

### 4. Start Ollama service

```bash
ollama serve
```

## 🎯 Usage

### Basic usage

```bash
# Run the main program
node src/index.js
```

### Interactive mode (Recommended)

```bash
# Run the advanced interactive system
node src/interactive-chat.js
```

### Enhanced context mode (Best experience)

```bash
# Run the enhanced context-aware system
node src/enhanced-context-chat.js
```

### Custom questions

Modify the question in `src/index.js`:

```js
const result = await chain.invoke('Your question');
```

### Add new documents

1. Place documents in the `src/sources/txt/` directory
2. Modify the file path in `src/index.js`:

```js
const loader = new TextLoader('./src/sources/txt/your-file.txt');
```

## 🎮 Interactive Commands

### Basic Commands (all modes)

-   `/help` - Show available commands
-   `/history` - Show conversation history
-   `/clear` - Clear conversation history
-   `/status` - Show system status
-   `/quit` or `/exit` - Exit the program

### Enhanced Commands (enhanced-context-chat.js only)

-   `/context` - Show current context summary
-   `/turn` - Show current conversation turn

### Example Enhanced Interactive Session

```
🤖 Enhanced Context Q&A System
💡 Type your question or use commands (type /help for commands)
📚 Available context: Situ Yongcong's profile
🧠 Enhanced context awareness enabled
======================================================================

❓ Your question: What is Situ Yongcong's expertise?

🤔 Thinking... (Turn 1)

🤖 Answer:
根据文档，司徒永聪的技术专长包括：

1. 系统运维（System Operations）
2. 前端开发（Front-end Development）
3. QNAP NAS 调优
4. 自动化脚本开发
5. 硬件诊断

他专注于技术实践，擅长从搭建博客到故障诊断，再到自动化脚本的各种技术领域。

--- 喵

❓ Your question: Tell me more about his blog

🤔 Thinking... (Turn 2)

🤖 Answer:
基于我们刚才关于司徒永聪技术专长的讨论，让我详细介绍一下他的博客：

司徒永聪经营个人博客 situ2001.com/blog，内容涵盖多个领域：
- 系统运维
- 前端开发
- 自动化脚本
- 硬件诊断
- 生产力工具
- 学习成长
- 旅行见闻

他通过博客分享技术实践与生活体验，体现了"动手且深入"的技术态度。

--- 喵

❓ Your question: /context

🧠 Current Context Summary:
========================================
Recent conversation focused on: technical expertise discussion, blog and online presence. We've discussed 2 topics so far.

📊 Conversation Statistics:
   Total turns: 2
   History entries: 2
```

## 📁 Project Structure

```
testing-langchainjs/
├── src/
│   ├── index.js              # Main program entry
│   ├── interactive-chat.js   # Advanced interactive system
│   ├── enhanced-context-chat.js # Enhanced context-aware system
│   ├── text-split.js         # Text splitting module
│   ├── embedding.js          # Vectorization module
│   ├── retrival-qa-chain.js  # Retrieval Q&A chain
│   ├── retrive.js           # Retrieval module
│   ├── comparison-demo.js    # Comparison demo
│   ├── retrieval-config-demo.js # Retrieval configuration demo
│   └── sources/
│       ├── txt/              # Text documents
│       └── pdf/              # PDF documents
├── package.json
└── README.md
```

## 🔧 Configuration Options

### Retrieval configuration

```js
vector_store.asRetriever({
    k: 3, // Return top 3 documents
    searchType: 'similarity', // Similarity search
});
```

### Text splitting configuration

```js
const textSplitter = new CharacterTextSplitter({
    chunkSize: 500, // Maximum characters per chunk
    chunkOverlap: 50, // Overlap between chunks
});
```

### Model configuration

```js
const llm = new Ollama({
    model: 'llama3',
    baseUrl: 'http://127.0.0.1:11434',
    temperature: 0,
});
```

## 🧠 Enhanced Context Features

### 1. Context Summary

The enhanced system automatically generates context summaries based on conversation topics:

-   Technical expertise discussions
-   Blog and online presence
-   Projects and work experience
-   Personal life and travel
-   General information

### 2. Conversation Turn Tracking

-   Each question is numbered as a "turn"
-   System maintains awareness of conversation progression
-   Better continuity between related questions

### 3. Extended History

-   Keeps last 8 conversation turns (vs 5 in basic version)
-   Includes turn numbers for better reference
-   Automatic context summary updates

### 4. Smart Prompt Engineering

```js
const prompt = PromptTemplate.fromTemplate(`
You are a helpful AI assistant with excellent memory and context awareness. You should maintain conversation continuity and build upon previous exchanges.

=== CONVERSATION CONTEXT ===
{context_summary}

=== PREVIOUS CONVERSATION ===
{history}

=== CURRENT SITUATION ===
- Turn: {turn}
- User's current question: {question}

=== RELEVANT DOCUMENTS ===
{docs}

=== INSTRUCTIONS ===
1. Answer the current question based on the provided documents
2. Consider the conversation history and context summary
3. If the question references previous topics, acknowledge and build upon them
4. If this is a follow-up question, reference the previous context appropriately
5. Maintain a conversational tone
6. Answer in Chinese and add "--- 喵" at the end

=== YOUR RESPONSE ===
`);
```

## 🆚 LCEL vs Traditional Approach

### LCEL approach (Recommended)

```js
const chain = RunnableSequence.from([
    {
        docs: vector_store.asRetriever({k: 3}),
        question: new RunnablePassthrough(),
    },
    PromptTemplate.fromTemplate('Answer: {question}\nContext: {docs}'),
    llm,
    new StringOutputParser(),
]);
```

### Traditional approach

```js
const combineChain = await createStuffDocumentsChain({llm, prompt});
const retrievalChain = await createRetrievalChain({
    combineDocsChain: combineChain,
    retriever: vector_store.asRetriever(3),
});
```

## 🎨 Advanced Features

### 1. Maximum Marginal Relevance (MMR) Search

```js
vector_store.asRetriever({
    k: 3,
    searchType: 'mmr',
    fetchK: 5,
    lambda: 0.5,
});
```

### 2. Similarity threshold filtering

```js
vector_store.asRetriever({
    k: 3,
    scoreThreshold: 0.7,
});
```

### 3. Custom prompt templates

```js
const prompt = PromptTemplate.fromTemplate(`
Answer the following question based on the documents: {question}

Relevant documents:
{docs}

Please answer in Chinese and add "--- Meow" at the end of your answer.
`);
```

### 4. Conversation history

The interactive system maintains conversation history and includes it in context for better continuity.

### 5. Context-aware responses

The enhanced system can:

-   Reference previous conversations
-   Build upon earlier topics
-   Maintain conversation flow
-   Provide more coherent multi-turn interactions

## 🐛 Common Issues

### Q: Encounter "model not found" error?

A: Make sure the corresponding Ollama model is installed:

```bash
ollama list  # View installed models
ollama pull llama3  # Install missing model
```

### Q: Encounter "invalid input type" error?

A: Make sure to use `fromTexts` instead of `fromDocuments`:

```js
// Correct
MemoryVectorStore.fromTexts(texts, metadata, embeddings);

// Wrong
MemoryVectorStore.fromDocuments(documents, embeddings);
```

### Q: Slow retrieval speed?

A: You can adjust retrieval parameters:

```js
vector_store.asRetriever({
    k: 2, // Reduce number of returned documents
    scoreThreshold: 0.8, // Increase similarity threshold
});
```

### Q: Poor answer quality?

A: Try using MMR search:

```js
vector_store.asRetriever({
    searchType: 'mmr',
    fetchK: 5,
    lambda: 0.5,
});
```

## 📊 Performance Optimization Tips

1.  **Document chunking**: Reasonably set `chunkSize` and `chunkOverlap`
2.  **Retrieval quantity**: Adjust `k` value according to needs
3.  **Model selection**: Choose appropriate models based on hardware configuration
4.  **Caching mechanism**: Consider adding vector caching to improve performance
5.  **Context management**: Use enhanced context mode for better conversation flow

## 🤝 Contributing

1.  Fork the project
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

-   [LangChain.js](https://js.langchain.com/) - Powerful LLM application framework
-   [Ollama](https://ollama.ai/) - Local large model runtime environment
-   [Chroma](https://www.trychroma.com/) - Vector database

## 📞 Contact

-   Author: Situ Yongcong
-   Blog: situ2001.com/blog
-   Email: 2369558390@qq.com

---

⭐ If this project helps you, please give it a star!
