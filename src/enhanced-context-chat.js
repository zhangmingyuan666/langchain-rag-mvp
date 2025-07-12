import { OllamaEmbeddings } from "@langchain/ollama";
import { Ollama } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from '@langchain/core/output_parsers';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CharacterTextSplitter } from '@langchain/textsplitters';
import readline from 'readline';

class EnhancedContextQASystem {
  constructor() {
    this.vectorStore = null;
    this.chain = null;
    this.conversationHistory = [];
    this.contextSummary = "";
    this.isInitialized = false;
    this.conversationTurn = 0;
  }

  async initialize() {
    console.log("🚀 Initializing Enhanced Context Q&A System...");
    
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

      // 5. 增强的 Prompt 模板
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

      // 6. LCEL Chain with enhanced context
      this.chain = RunnableSequence.from([
        {
          docs: this.vectorStore.asRetriever({
            k: 3,
            searchType: "similarity"
          }),
          question: new RunnablePassthrough(),
          history: () => this.getConversationHistory(),
          context_summary: () => this.contextSummary,
          turn: () => this.conversationTurn
        },
        prompt,
        llm,
        new StringOutputParser()
      ]);

      this.isInitialized = true;
      console.log("✅ Enhanced system initialized successfully!");
      
    } catch (error) {
      console.error("❌ Initialization failed:", error.message);
      throw error;
    }
  }

  getConversationHistory() {
    if (this.conversationHistory.length === 0) return "No previous conversation.";
    
    return this.conversationHistory
      .slice(-8) // 保留最近8轮对话，增加上下文
      .map((entry, index) => {
        const turnNumber = this.conversationHistory.length - 8 + index + 1;
        return `[Turn ${turnNumber}] Q: ${entry.question}\nA: ${entry.answer}`;
      })
      .join('\n\n');
  }

  updateContextSummary() {
    if (this.conversationHistory.length === 0) {
      this.contextSummary = "This is the beginning of our conversation.";
      return;
    }

    // 基于最近的对话生成上下文摘要
    const recentConversations = this.conversationHistory.slice(-3);
    const topics = recentConversations.map(entry => {
      const question = entry.question.toLowerCase();
      if (question.includes('expertise') || question.includes('skill') || question.includes('技术')) {
        return 'technical expertise discussion';
      } else if (question.includes('blog') || question.includes('website') || question.includes('博客')) {
        return 'blog and online presence';
      } else if (question.includes('project') || question.includes('work') || question.includes('项目')) {
        return 'projects and work experience';
      } else if (question.includes('travel') || question.includes('life') || question.includes('生活')) {
        return 'personal life and travel';
      } else {
        return 'general information';
      }
    });

    const uniqueTopics = [...new Set(topics)];
    this.contextSummary = `Recent conversation focused on: ${uniqueTopics.join(', ')}. We've discussed ${this.conversationHistory.length} topics so far.`;
  }

  async askQuestion(question) {
    if (!this.isInitialized) {
      throw new Error("System not initialized");
    }

    this.conversationTurn++;
    
    try {
      const result = await this.chain.invoke(question);
      
      // 保存对话历史
      this.conversationHistory.push({
        question,
        answer: result,
        timestamp: new Date().toISOString(),
        turn: this.conversationTurn
      });

      // 更新上下文摘要
      this.updateContextSummary();

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
  /context  - Show current context summary
  /clear    - Clear conversation history
  /status   - Show system status
  /turn     - Show current conversation turn
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
    console.log("=" .repeat(60));
    this.conversationHistory.forEach((entry, index) => {
      console.log(`\n[Turn ${entry.turn}] Q: ${entry.question}`);
      console.log(`   A: ${entry.answer.substring(0, 120)}...`);
      console.log(`   Time: ${new Date(entry.timestamp).toLocaleString()}`);
    });
  }

  showContext() {
    console.log("\n🧠 Current Context Summary:");
    console.log("=" .repeat(40));
    console.log(this.contextSummary);
    console.log(`\n📊 Conversation Statistics:`);
    console.log(`   Total turns: ${this.conversationTurn}`);
    console.log(`   History entries: ${this.conversationHistory.length}`);
  }

  showStatus() {
    console.log(`
🔧 System Status:
  Initialized: ${this.isInitialized ? '✅ Yes' : '❌ No'}
  Vector Store: ${this.vectorStore ? '✅ Ready' : '❌ Not ready'}
  Chain: ${this.chain ? '✅ Ready' : '❌ Not ready'}
  Conversation Turn: ${this.conversationTurn}
  History Entries: ${this.conversationHistory.length}
  Context Summary: ${this.contextSummary ? '✅ Active' : '❌ None'}
    `);
  }

  clearHistory() {
    this.conversationHistory = [];
    this.contextSummary = "";
    this.conversationTurn = 0;
    console.log("🗑️  Conversation history and context cleared.");
  }

  async startInteractiveMode() {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    console.log("\n🤖 Enhanced Context Q&A System");
    console.log("💡 Type your question or use commands (type /help for commands)");
    console.log("📚 Available context: Situ Yongcong's profile");
    console.log("🧠 Enhanced context awareness enabled");
    console.log("=" .repeat(70));

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
            case '/context':
              this.showContext();
              break;
            case '/clear':
              this.clearHistory();
              break;
            case '/status':
              this.showStatus();
              break;
            case '/turn':
              console.log(`\n🔄 Current conversation turn: ${this.conversationTurn}`);
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
          console.log(`\n🤔 Thinking... (Turn ${this.conversationTurn + 1})`);
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
  const qaSystem = new EnhancedContextQASystem();
  
  try {
    await qaSystem.initialize();
    await qaSystem.startInteractiveMode();
  } catch (error) {
    console.error("💥 Application failed to start:", error.message);
    process.exit(1);
  }
}

main();