import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CharacterTextSplitter } from 'langchain/text_splitter';

async function splitTextToChunk() {
    const loader = new TextLoader('./src/sources/txt/situ-profile.txt');
    const docs = await loader.load();

    const textSplitter = new CharacterTextSplitter({
        chunkSize: 500,       // 减小每块最大字符数
        chunkOverlap: 50,     // 减小块与块之间重叠字符数
    });

    const chunks = await textSplitter.splitDocuments(docs);  // 拆分文档块:contentReference[oaicite:6]{index=6}

    return chunks;
}

export {
    splitTextToChunk
}