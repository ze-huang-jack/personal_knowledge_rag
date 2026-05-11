"""
RAGAS evaluation script.

Run after indexing some documents:
    python evaluate.py

The script uses hardcoded test questions — edit them to match your documents.
For best results, add ground_truth answers to enable all metrics.
"""
import json
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall

load_dotenv()

CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "knowledge_base"


def get_llm():
    import os
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek/deepseek-v4-flash"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )


def get_embeddings():
    import os
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )


SYSTEM_PROMPT = (
    "你是一个知识库助手。根据以下检索到的上下文回答问题。"
    "如果上下文中没有答案，就说不知道，不要编造。\n\n"
    "上下文：\n{context}"
)


def load_test_questions() -> list[dict]:
    """Load test questions. Edit this to match your documents."""
    # Default test set — replace with questions relevant to your documents
    return [
        {
            "question": "这份文档的主要内容是什么？",
            "ground_truth": None,
        },
        {
            "question": "文档中提到了哪些关键概念？",
            "ground_truth": None,
        },
        {
            "question": "总结文档的核心观点。",
            "ground_truth": None,
        },
    ]


def run_evaluation(test_questions: list[dict]):
    """Run RAGAS evaluation over test questions."""
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )

    collection = vectorstore._collection
    if collection.count() == 0:
        print("❌ Chroma 中没有文档。请先上传并索引文档。")
        sys.exit(1)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    def _format_docs(docs):
        return "\n\n".join(f"[{d.metadata.get('file_name', '')}]\n{d.page_content}" for d in docs)

    chain = prompt | llm | StrOutputParser()

    samples = []
    results = []

    print(f"📊 运行 {len(test_questions)} 个测试问题...\n")

    for i, q in enumerate(test_questions):
        question = q["question"]
        ground_truth = q.get("ground_truth")

        print(f"Q{i+1}: {question}")
        docs = retriever.invoke(question)
        context_text = _format_docs(docs)
        answer = chain.invoke({"context": context_text, "input": question})
        contexts = [doc.page_content for doc in docs]

        print(f"   Answer: {answer[:150]}...")
        print(f"   Contexts retrieved: {len(contexts)}")

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        samples.append(sample)
        results.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "context_count": len(contexts),
        })
        print()

    # Metrics that don't require ground_truth
    metrics = [Faithfulness(), ResponseRelevancy(), ContextPrecision()]
    if any(q.get("ground_truth") for q in test_questions):
        metrics.append(ContextRecall())

    print("🔬 计算 RAGAS 指标...")
    dataset = EvaluationDataset(samples=samples)
    eval_result = evaluate(dataset=dataset, metrics=metrics)

    print("\n" + "=" * 50)
    print("📈 Evaluation Results")
    print("=" * 50)
    for metric_name, score in eval_result.items():
        print(f"  {metric_name}: {score:.4f}")

    # Write detailed report
    report_path = Path("./eval_report.json")
    report = {
        "metrics": {k: float(v) for k, v in eval_result.items()},
        "details": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 详细报告已写入 {report_path}")

    return eval_result


if __name__ == "__main__":
    test_questions = load_test_questions()
    run_evaluation(test_questions)
