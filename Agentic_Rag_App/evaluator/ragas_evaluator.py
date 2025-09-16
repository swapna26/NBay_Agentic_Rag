"""RAGAs-based evaluation system for RAG implementation."""

import asyncio
import json
import pandas as pd
import structlog
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from datasets import Dataset
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from config import config
from utils.database_client import DatabaseClient
from utils.rag_client import RAGClient

logger = structlog.get_logger()


class RAGASEvaluator:
    """RAGAs-based evaluator for RAG system."""

    def __init__(self):
        self.db_client = DatabaseClient()
        self.rag_client = RAGClient()

        # Initialize Ollama models for RAGAs
        self.embedding_model = OllamaEmbedding(
            model_name=config.embedding_model,
            base_url=config.ollama_base_url
        )

        self.llm = Ollama(
            model=config.llm_model,
            base_url=config.ollama_base_url,
            request_timeout=120.0
        )

        # RAGAs metrics mapping
        self.metrics_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness
        }

    async def initialize(self):
        """Initialize the evaluator components."""
        try:
            await self.db_client.initialize()

            # Check if backend is healthy
            if not await self.rag_client.health_check():
                logger.warning("Backend API is not healthy, some features may not work")

            logger.info("RAGAs evaluator initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize evaluator", error=str(e))
            return False

    async def load_test_dataset(self, dataset_path: str = None) -> List[Dict[str, Any]]:
        """Load test dataset from JSON file."""
        path = Path(dataset_path or config.test_dataset_path)

        if not path.exists():
            logger.warning("Test dataset not found, generating new one", path=str(path))
            return await self.generate_test_dataset()

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both formats: direct list or wrapped in {"questions": [...]}
            if isinstance(data, list):
                dataset = data
            elif isinstance(data, dict) and "questions" in data:
                dataset = data["questions"]
            else:
                logger.error("Invalid dataset format", data_type=type(data).__name__)
                return []

            logger.info("Loaded test dataset", path=str(path), count=len(dataset))
            return dataset

        except Exception as e:
            logger.error("Failed to load test dataset", error=str(e), path=str(path))
            return []

    async def generate_test_dataset(self, num_questions: int = None) -> List[Dict[str, Any]]:
        """Generate test questions based on indexed documents."""
        num_questions = num_questions or config.num_test_questions

        try:
            # Get sample documents from the database
            sample_docs = await self.db_client.get_sample_documents(limit=min(num_questions, 20))

            if not sample_docs:
                logger.error("No documents found in database for test generation")
                return []

            test_questions = []

            for i, doc in enumerate(sample_docs[:num_questions]):
                # Generate question based on document content
                question_prompt = f"""Based on the following document content, generate a specific question that can be answered using this information:

Document: {doc['content'][:1000]}

Generate a clear, specific question that would require understanding this document to answer correctly. Return only the question, nothing else."""

                try:
                    question_response = self.llm.complete(question_prompt)
                    question = str(question_response).strip()

                    if question and len(question.split()) >= 5:  # Ensure meaningful question
                        test_questions.append({
                            "question": question,
                            "source_document": doc['source_file'],
                            "expected_context": doc['content'][:500]  # First 500 chars as expected context
                        })

                        logger.info(f"Generated question {i+1}/{num_questions}", question=question[:100])

                except Exception as e:
                    logger.warning(f"Failed to generate question for document {i+1}", error=str(e))
                    continue

            # Save generated dataset
            await self.save_test_dataset(test_questions)

            logger.info("Generated test dataset", count=len(test_questions))
            return test_questions

        except Exception as e:
            logger.error("Failed to generate test dataset", error=str(e))
            return []

    async def save_test_dataset(self, dataset: List[Dict[str, Any]]):
        """Save test dataset to JSON file."""
        try:
            path = Path(config.test_dataset_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

            logger.info("Saved test dataset", path=str(path), count=len(dataset))

        except Exception as e:
            logger.error("Failed to save test dataset", error=str(e))

    async def collect_rag_responses(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect RAG responses for evaluation questions."""
        responses = []

        for i, q_data in enumerate(questions):
            try:
                question = q_data["question"]

                # Query the RAG system (use fast mode for evaluation)
                rag_response = await self.rag_client.query(question, use_agents=False)

                if "error" in rag_response:
                    logger.warning(f"RAG query failed for question {i+1}", error=rag_response["error"])
                    continue

                response_data = {
                    "question": question,
                    "answer": rag_response.get("answer", ""),
                    "contexts": [ctx.get("content", "") for ctx in rag_response.get("context", [])],
                    "ground_truth": q_data.get("expected_context", ""),
                    "source_document": q_data.get("source_document", "")
                }

                responses.append(response_data)

                logger.info(f"Collected response {i+1}/{len(questions)}",
                          question=question[:50],
                          answer_length=len(response_data["answer"]),
                          contexts_count=len(response_data["contexts"]))

            except Exception as e:
                logger.error(f"Failed to collect response for question {i+1}", error=str(e))
                continue

        return responses

    async def evaluate_with_ragas(self, responses: List[Dict[str, Any]],
                                selected_metrics: List[str] = None) -> Dict[str, Any]:
        """Evaluate responses using RAGAs metrics."""
        try:
            if not responses:
                logger.error("No responses to evaluate")
                return {}

            # Prepare data for RAGAs
            eval_data = {
                "question": [r["question"] for r in responses],
                "answer": [r["answer"] for r in responses],
                "contexts": [r["contexts"] for r in responses],
                "ground_truth": [r["ground_truth"] for r in responses]
            }

            # Create dataset
            dataset = Dataset.from_dict(eval_data)

            # Select metrics to evaluate
            metrics_to_use = selected_metrics or config.metrics
            active_metrics = []

            for metric_name in metrics_to_use:
                if metric_name in self.metrics_map:
                    active_metrics.append(self.metrics_map[metric_name])
                else:
                    logger.warning(f"Unknown metric: {metric_name}")

            if not active_metrics:
                logger.error("No valid metrics selected")
                return {}

            logger.info("Starting RAGAs evaluation",
                       responses_count=len(responses),
                       metrics=metrics_to_use)

            # Configure RAGAs with Ollama models
            # Note: RAGAs might need specific configuration for Ollama
            result = evaluate(
                dataset=dataset,
                metrics=active_metrics,
                llm=self.llm,
                embeddings=self.embedding_model
            )

            # Convert results to dict
            evaluation_results = {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(responses),
                "metrics_evaluated": metrics_to_use,
                "scores": {}
            }

            # Extract metric scores
            for metric_name in metrics_to_use:
                if metric_name in result:
                    score = result[metric_name]
                    if isinstance(score, (list, tuple)):
                        evaluation_results["scores"][metric_name] = {
                            "mean": float(pd.Series(score).mean()),
                            "std": float(pd.Series(score).std()),
                            "min": float(pd.Series(score).min()),
                            "max": float(pd.Series(score).max()),
                            "individual_scores": [float(s) for s in score]
                        }
                    else:
                        evaluation_results["scores"][metric_name] = float(score)

            logger.info("RAGAs evaluation completed",
                       metrics_count=len(evaluation_results["scores"]))

            return evaluation_results

        except Exception as e:
            logger.error("RAGAs evaluation failed", error=str(e))
            return {"error": str(e)}

    async def generate_evaluation_report(self, results: Dict[str, Any],
                                       output_path: str = None) -> str:
        """Generate comprehensive evaluation report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"rag_evaluation_report_{timestamp}.json"
            report_path = Path(config.reports_dir) / report_filename

            if output_path:
                report_path = Path(output_path)

            # Add system information
            system_info = await self.rag_client.get_system_info()
            db_stats = {
                "document_count": await self.db_client.get_document_count(),
                "chunk_count": await self.db_client.get_chunk_count(),
                "source_files": await self.db_client.get_all_source_files()
            }

            comprehensive_report = {
                "evaluation_metadata": {
                    "timestamp": results.get("timestamp", datetime.now().isoformat()),
                    "evaluator_version": "1.0.0",
                    "ragas_version": "0.1.0",  # Update with actual version
                    "configuration": {
                        "embedding_model": config.embedding_model,
                        "llm_model": config.llm_model,
                        "top_k": config.top_k,
                        "chunk_size": config.chunk_size
                    }
                },
                "system_info": system_info,
                "database_stats": db_stats,
                "evaluation_results": results,
                "summary": self._generate_summary(results)
            }

            # Save comprehensive report
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)

            logger.info("Generated evaluation report", path=str(report_path))
            return str(report_path)

        except Exception as e:
            logger.error("Failed to generate evaluation report", error=str(e))
            return ""

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of evaluation results."""
        summary = {
            "overall_performance": "unknown",
            "best_metric": None,
            "worst_metric": None,
            "recommendations": []
        }

        try:
            if "scores" not in results:
                return summary

            scores = results["scores"]
            metric_averages = {}

            # Calculate average scores
            for metric, score_data in scores.items():
                if isinstance(score_data, dict) and "mean" in score_data:
                    metric_averages[metric] = score_data["mean"]
                else:
                    metric_averages[metric] = float(score_data)

            if metric_averages:
                # Find best and worst metrics
                summary["best_metric"] = max(metric_averages, key=metric_averages.get)
                summary["worst_metric"] = min(metric_averages, key=metric_averages.get)

                # Overall performance assessment
                overall_avg = sum(metric_averages.values()) / len(metric_averages)
                if overall_avg >= 0.8:
                    summary["overall_performance"] = "excellent"
                elif overall_avg >= 0.6:
                    summary["overall_performance"] = "good"
                elif overall_avg >= 0.4:
                    summary["overall_performance"] = "fair"
                else:
                    summary["overall_performance"] = "poor"

                # Generate recommendations
                summary["recommendations"] = self._generate_recommendations(metric_averages)

        except Exception as e:
            logger.error("Failed to generate summary", error=str(e))

        return summary

    def _generate_recommendations(self, metric_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on metric scores."""
        recommendations = []

        try:
            for metric, score in metric_scores.items():
                if score < 0.5:
                    if metric == "faithfulness":
                        recommendations.append("Consider improving context relevance and reducing hallucinations in responses")
                    elif metric == "answer_relevancy":
                        recommendations.append("Improve query understanding and response relevance")
                    elif metric == "context_precision":
                        recommendations.append("Enhance retrieval quality to get more relevant context")
                    elif metric == "context_recall":
                        recommendations.append("Increase context retrieval coverage and chunk size")
                    elif metric == "answer_similarity":
                        recommendations.append("Improve answer generation quality and consistency")
                    elif metric == "answer_correctness":
                        recommendations.append("Focus on factual accuracy and completeness of answers")

            if not recommendations:
                recommendations.append("Overall performance is good. Consider fine-tuning for specific use cases.")

        except Exception as e:
            logger.error("Failed to generate recommendations", error=str(e))

        return recommendations

    async def run_complete_evaluation(self, custom_questions: List[str] = None,
                                    metrics: List[str] = None) -> str:
        """Run complete evaluation pipeline."""
        try:
            logger.info("Starting complete RAG evaluation")

            # Load or generate test dataset
            if custom_questions:
                test_data = [{"question": q, "source_document": "custom", "expected_context": ""}
                           for q in custom_questions]
            else:
                test_data = await self.load_test_dataset()

            if not test_data:
                logger.error("No test data available")
                return ""

            # Collect RAG responses
            responses = await self.collect_rag_responses(test_data)

            if not responses:
                logger.error("No responses collected")
                return ""

            # Evaluate with RAGAs
            evaluation_results = await self.evaluate_with_ragas(responses, metrics)

            if not evaluation_results or "error" in evaluation_results:
                logger.error("Evaluation failed", error=evaluation_results.get("error"))
                return ""

            # Generate report
            report_path = await self.generate_evaluation_report(evaluation_results)

            if report_path:
                logger.info("Complete evaluation finished successfully", report=report_path)
                return report_path
            else:
                logger.error("Failed to generate evaluation report")
                return ""

        except Exception as e:
            logger.error("Complete evaluation failed", error=str(e))
            return ""

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.db_client.cleanup()
            await self.rag_client.cleanup()
            logger.info("Evaluator cleanup completed")
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))