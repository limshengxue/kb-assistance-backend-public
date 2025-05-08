import asyncio
import chromadb
import os
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator
)
from llama_index.core.schema import TextNode
import pandas as pd
from retriever import Retriever
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.evaluation.retrieval.evaluator import RetrievalEvalMode


eval_dataset_file_path = "qa_evaluation_dataset.json"

def get_document_nodes():
    """Retrieve documents from ChromaDB collection and convert to LlamaIndex nodes with persistent node_ids."""
    db = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "digicelcom_documents"
    chroma_collection = db.get_collection(collection_name)
    results = chroma_collection.get(include=["documents", "metadatas", "embeddings"])

    nodes = []

    if results and results["documents"]:
        documents = results["documents"]
        metadatas = results.get("metadatas", [{}] * len(documents))
        ids = results.get("ids", [f"node-{i}" for i in range(len(documents))])

        for i, (doc, metadata, node_id) in enumerate(zip(documents, metadatas, ids)):
            if doc:
                node = TextNode(
                    id_=node_id,  # <--- IMPORTANT: preserve node ID
                    text=doc,
                    metadata=metadata or {"index": i},
                )
                nodes.append(node)
    else:
        print("No documents found in the collection")

    return nodes
    


def generate_evaluation_dataset(nodes, num_questions_per_chunk=5, save_path=eval_dataset_file_path):
    """Generate evaluation questions based on retrieved documents."""
    if not nodes:
        print("No nodes available for evaluation")
        return None
    
    # Initialize LLM
    llm = OpenAI(
        model="gpt-3.5-turbo", 
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.2
    )
    
    print(f"\nGenerating evaluation questions for {len(nodes)} documents...")
    
    # Generate question-context pairs
    qa_dataset = generate_question_context_pairs(
        nodes,
        llm=llm,
        num_questions_per_chunk=num_questions_per_chunk,
        )
    
    # Save questions to file
    qa_dataset.save_json(save_path)
    
    print(f"Successfully generated {len(qa_dataset.queries)} evaluation questions")
    return qa_dataset

async def run_evaluation():
    """Run evaluation on generated questions."""
    # Initialize Retriever
    retriever = Retriever()
    await retriever.initialize()

    # Initialize Evaluation Dataset
    document_nodes = get_document_nodes()

    # load qa dataset
    if os.path.exists(eval_dataset_file_path):
        qa_dataset = EmbeddingQAFinetuneDataset.from_json(eval_dataset_file_path)
        print("Load existing evaluation dataset")
    else:
        # Generate evaluation questions
        qa_dataset = generate_evaluation_dataset(document_nodes)
    
    # Evaluate
    index = retriever.get_index()
    mistralai_embedding_hybrid_search_retriever = retriever.get_retriever()

    await evaluate_retriever(mistralai_embedding_hybrid_search_retriever, qa_dataset, "MistralAI Embedding Hybrid Search (Cutoff = 0.6)")

async def evaluate_retriever(retriever, qa_dataset, retriever_name, cutoff=0.6):
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], 
        retriever=retriever, 
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=cutoff)]
    )
    eval_results = []
    mode = RetrievalEvalMode.from_str(qa_dataset.mode)

    for query_id, query in qa_dataset.queries.items():
        expected_ids = qa_dataset.relevant_docs[query_id]
        print(expected_ids)
        eval_result = await retriever_evaluator.aevaluate(query, expected_ids=expected_ids, mode=mode)
        eval_results.append(eval_result)
        # sleep to avoid rate limit
        await asyncio.sleep(30)

    # save results
    save_results(retriever_name, eval_results)
    

def save_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
    )

    # save
    if os.path.isfile("retrieve_results.csv"):
        metric_df.to_csv("retrieve_results.csv", mode='a', header=False, index=False)
    else:
        metric_df.to_csv("retrieve_results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(run_evaluation())    
    
    



