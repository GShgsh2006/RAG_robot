﻿﻿﻿﻿﻿from ragas import evaluate
from ragas import SingleTurnSample
from ragas import EvaluationDataset
from app import context_hanshu
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
import logging
from langchain_community.chat_models import ChatOllama
from ai import get_ai_response

def evaluate_ragas(query):
    try:
        Context, retrieved_knowledge, final_contexts = context_hanshu(query)
        
        ai_response = get_ai_response(query)
        
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=final_contexts,
            response=ai_response,
        )

        dataset = EvaluationDataset(
            samples=[sample]
        )

        llm = ChatOllama(
            model="llama3"
        )

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=llm,
            batch_size=32
        )

        logging.info(f"RAGAS璇勪及缁撴灉锛歿result}")
        return result

    except Exception as e:
        logging.error(f"RAGAS璇勪及澶辫触锛歿str(e)}")
        return None
