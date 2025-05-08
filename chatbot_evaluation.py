import asyncio
import os
import time
from chatbot import Chatbot
from retriever import Retriever
from retriever_evaluation import generate_evaluation_dataset, get_document_nodes
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner
)
from llama_index.llms.openai import OpenAI
import pandas as pd


eval_dataset_file_path = "qa_evaluation_dataset.json"

class ChatEngineEvaluator:
    def __init__(self):
        gpt4 = OpenAI(temperature=0, model="gpt-4o")
        self.faithful_evaluator = FaithfulnessEvaluator(gpt4)
        self.relevancy_evaluator = RelevancyEvaluator(gpt4)


    async def run_evaluation(self):
        """Run evaluation on generated questions."""
        # Initialize Retriever
        retriever = Retriever()
        await retriever.initialize()

        # Get Query Engine
        chatbot = Chatbot(retriever)
        chat_engine = chatbot.get_chat_engine()

        # Initialize Evaluation Dataset
        document_nodes = get_document_nodes()

        # load qa dataset
        if os.path.exists(eval_dataset_file_path):
            qa_dataset = EmbeddingQAFinetuneDataset.from_json(eval_dataset_file_path)
            print("Load existing evaluation dataset")
        else:
            # Generate evaluation questions
            qa_dataset = generate_evaluation_dataset(document_nodes)

        queries = list(qa_dataset.queries.values())

        # Evaluate
        eval_results = await self.evaluate_chat_engine(chat_engine, queries)

        self.save_results("OpenAI Response Generator", eval_results)

    def get_responses(self, chat_engine, queries):
        responses = []
        for query in queries:
            response = chat_engine.chat(query)
            responses.append(response)
            # add delay to avoid rate limit
            time.sleep(30)
        return responses

    async def evaluate_chat_engine(self, chat_engine, queries):
        responses = self.get_responses(chat_engine, queries)
        runner = BatchEvalRunner(
            {"faithfulness": self.faithful_evaluator, "relevancy": self.relevancy_evaluator},
        )
        
        # Compute evaluation
        eval_results = await runner.aevaluate_responses(
            responses=responses, queries=queries
        )

        return eval_results

    def save_results(self, name, eval_results):
        faithfulness_score = sum(result.passing for result in eval_results['faithfulness']) / len(eval_results['faithfulness'])
        relevancy_score = sum(result.passing for result in eval_results['relevancy']) / len(eval_results['relevancy'])

        metric_df = pd.DataFrame(
            {"Retriever Name": [name], "Faithfulness": [faithfulness_score], 
             "Relevancy": [relevancy_score], "Timestamp": [pd.Timestamp.now()]}
        )

        # save
        if os.path.isfile("response_results.csv"):
            metric_df.to_csv("response_results.csv", mode='a', header=False, index=False)
        else:
            metric_df.to_csv("response_results.csv", index=False)



evaluator = ChatEngineEvaluator()
asyncio.run(evaluator.run_evaluation())    


