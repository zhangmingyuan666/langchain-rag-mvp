import { OllamaEmbeddings } from "@langchain/ollama";
import { Ollama } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from '@langchain/core/output_parsers';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CharacterTextSplitter } from '@langchain/textsplitters';
import readline from 'readline';

class InteractiveQASystem {
  constructor() {
    this.vectorStore = null;
    this.chain = null;
    this.conversationHistory = [];
    this.isInitialized = false;
  }

  async initialize() {
    console.log("🚀 Initializing Q&A System...");
    
    try {
      // 1. 嵌入模型
      const embed_model = new OllamaEmbeddings({ model: "mxbai-embed-large" });

      // 2. 加载和分割文档
      console.log("📖 Loading documents...");
      const loader = new TextLoader("./src/sources/txt/situ-profile.txt");
      const docs = await loader.load();

      const splitter = new CharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 50,
      });
      const chunks = await splitter.splitDocuments(docs);
      console.log(`📄 Loaded ${chunks.length} document chunks`);

      // 3. 构建向量数据库
      console.log("🔍 Building vector database...");
      this.vectorStore = await MemoryVectorStore.fromTexts(
        chunks.map(chunk => chunk.pageContent),
        chunks.map(chunk => chunk.metadata),
        embed_model
      );

      // 4. LLM
      const llm = new Ollama({
        model: "llama3",
        baseUrl: "http://127.0.0.1:11434"
      });

      // 5. Prompt with conversation history
      const prompt = PromptTemplate.fromTemplate(`
You are a helpful AI assistant. Answer the following question based on the provided context.

Previous conversation:
{history}

Current question: {question}

Relevant documents:
{docs}

Please answer in Chinese and add "--- 喵" at the end of your answer.
`);

      // 6. LCEL Chain
      this.chain = RunnableSequence.from([
        {
          docs: this.vectorStore.asRetriever({
            k: 3,
            searchType: "similarity"
          }),
          question: new RunnablePassthrough(),
          history: () => this.getConversationHistory()
        },
        prompt,
        llm,
        new StringOutputParser()
      ]);

      this.isInitialized = true;
      console.log("✅ System initialized successfully!");
      
    } catch (error) {
      console.error("❌ Initialization failed:", error.message);
      throw error;
    }
  }

  getConversationHistory() {
    if (this.conversationHistory.length === 0) return "No previous conversation.";
    
    return this.conversationHistory
      .slice(-5) // 只保留最近5轮对话
      .map(entry => `Q: ${entry.question}\nA: ${entry.answer}`)
      .join('\n\n');
  }

  async askQuestion(question) {
    if (!this.isInitialized) {
      throw new Error("System not initialized");
    }

    try {
      const result = await this.chain.invoke(question);
      
      // 保存对话历史
      this.conversationHistory.push({
        question,
        answer: result,
        timestamp: new Date().toISOString()
      });

      return result;
    } catch (error) {
      throw error;
    }
  }

  showHelp() {
    console.log(`
📋 Available Commands:
  /help     - Show this help message
  /history  - Show conversation history
  /clear    - Clear conversation history
  /status   - Show system status
  /quit     - Exit the program
  /exit     - Exit the program
    `);
  }

  showHistory() {
    if (this.conversationHistory.length === 0) {
      console.log("📝 No conversation history yet.");
      return;
    }

    console.log("\n📝 Conversation History:");
    console.log("=" .repeat(50));
    this.conversationHistory.forEach((entry, index) => {
      console.log(`\n${index + 1}. Q: ${entry.question}`);
      console.log(`   A: ${entry.answer.substring(0, 100)}...`);
      console.log(`   Time: ${new Date(entry.timestamp).toLocaleString()}`);
    });
  }

  showStatus() {
    console.log(`
🔧 System Status:
  Initialized: ${this.isInitialized ? '✅ Yes' : '❌ No'}
  Vector Store: ${this.vectorStore ? '✅ Ready' : '❌ Not ready'}
  Chain: ${this.chain ? '✅ Ready' : '❌ Not ready'}
  Conversation History: ${this.conversationHistory.length} entries
    `);
  }

  clearHistory() {
    this.conversationHistory = [];
    console.log("🗑️  Conversation history cleared.");
  }

  async startInteractiveMode() {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    console.log("\n🤖 Interactive Q&A System");
    console.log("💡 Type your question or use commands (type /help for commands)");
    console.log("📚 Available context: Situ Yongcong's profile");
    console.log("=" .repeat(60));

    const askQuestion = () => {
      rl.question('\n❓ Your question: ', async (input) => {
        const trimmedInput = input.trim();

        if (trimmedInput === '') {
          console.log('⚠️  Please enter a question or command.');
          askQuestion();
          return;
        }

        // 处理命令
        if (trimmedInput.startsWith('/')) {
          const command = trimmedInput.toLowerCase();
          
          switch (command) {
            case '/help':
              this.showHelp();
              break;
            case '/history':
              this.showHistory();
              break;
            case '/clear':
              this.clearHistory();
              break;
            case '/status':
              this.showStatus();
              break;
            case '/quit':
            case '/exit':
              console.log('\n👋 Goodbye!');
              rl.close();
              return;
            default:
              console.log('❓ Unknown command. Type /help for available commands.');
          }
          
          askQuestion();
          return;
        }

        // 处理问题
        try {
          console.log('\n🤔 Thinking...');
          const result = await this.askQuestion(trimmedInput);
          console.log('\n🤖 Answer:');
          console.log(result);
        } catch (error) {
          console.error('\n❌ Error:', error.message);
        }

        askQuestion();
      });
    };

    askQuestion();
  }
}

// 启动应用
async function main() {
  const qaSystem = new InteractiveQASystem();
  
  try {
    await qaSystem.initialize();
    await qaSystem.startInteractiveMode();
  } catch (error) {
    console.error("💥 Application failed to start:", error.message);
    process.exit(1);
  }
}

main(); 