from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd
import yaml
import math

from src.pdf_parsing import PDFParser
from src.parsed_documents_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
# from src.tables_serialization import TableSerializer
from src.url_parsing import URLParser

from src.api_requests import APIProcessor
from src.retrieval import HybridRetriever


@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "qa.yaml",
                 pdf_reports_dir_name: str = "pdf_docs", serialized: bool = False, config_suffix: str = ""):
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name

        self.answers_file_path = root_path / f"answers{config_suffix}.json"
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"

        self.vector_db_dir = self.databases_path / "vector_db"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_db"

        self.parsed_documents_dirname = "01_parsed_documents"
        self.parsed_documents_debug_dirname = "01_parsed_documents_debug"
        self.merged_documents_dirname = f"02_merged_documents{suffix}"
        self.documents_markdown_dirname = f"03_documents_markdown{suffix}"

        self.parsed_documents_path = self.debug_data_path / self.parsed_documents_dirname
        self.parsed_documents_debug_path = self.debug_data_path / self.parsed_documents_debug_dirname
        self.merged_documents_path = self.debug_data_path / self.merged_documents_dirname
        self.documents_markdown_path = self.debug_data_path / self.documents_markdown_dirname


@dataclass
class RunConfig:
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 15
    top_n_retrieval: int = 7
    parallel_requests: int = 10
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = "openai"
    answering_model: str = "gpt-4o-mini-2024-07-18"  # or "gpt-4o-2024-08-06"
    config_suffix: str = ""


class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "qa.yaml",
                 pdf_reports_dir_name: str = "pdf_docs", run_config: RunConfig = RunConfig()):
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str,
                          pdf_reports_dir_name: str) -> PipelineConfig:
        """Initialize paths configuration based on run config settings"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )


    # Docling automatically downloads some models from huggingface when first used
    # I wanted to download them prior to running the pipeline and created this crutch
    @staticmethod
    def download_docling_models():
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_documents_sequential(self):
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_documents_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_documents_debug_path

        pdf_parser.parse_and_export(doc_dir=self.paths.pdf_reports_dir)
        print(f"PDF documents parsed and saved to {self.paths.parsed_documents_path}")

    def parse_pdf_documents_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """Parse PDF reports in parallel using multiple processes.

        Args:
            chunk_size: Number of PDFs to process in each worker
            num_workers: Number of parallel worker processes to use
        """
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_documents_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_documents_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))

        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF documents parsed and saved to {self.paths.parsed_documents_path}")

    # def serialize_tables(self, max_workers: int = 10):
    #     """Process tables in parsed documents using parallel threading"""
    #     serializer = TableSerializer()
    #     serializer.process_directory_parallel(
    #         self.paths.parsed_documents_path,
    #         max_workers=max_workers
    #     )

    def merge_documents(self):
        """Merge complex JSON documents into a simpler structure with a list of pages, where all text blocks are combined into a single string."""
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        _ = ptp.process_documents(
            input_dir=self.paths.parsed_documents_path,
            output_dir=self.paths.merged_documents_path
        )
        print(f"Documents saved to {self.paths.merged_documents_path}")

    def export_documents_to_markdown(self):
        """Export processed documents to markdown format for review."""
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        ptp.export_to_markdown(
            documents_dir=self.paths.parsed_documents_path,
            output_dir=self.paths.documents_markdown_path
        )
        print(f"Documents saved to {self.paths.documents_markdown_path}")

    def chunk_documents(self, include_serialized_tables: bool = False):
        """Split processed documents into smaller chunks for better processing."""
        text_splitter = TextSplitter()

        serialized_tables_dir = None
        if include_serialized_tables:
            serialized_tables_dir = self.paths.parsed_documents_path

        text_splitter.split_all_documents(
            self.paths.merged_documents_path,
            self.paths.documents_dir,
            serialized_tables_dir
        )
        print(f"Chunked documents saved to {self.paths.documents_dir}")

    def create_vector_db(self):
        """Create a vector database from all chunked documents."""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir

        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_documents(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")

    def create_bm25_db(self):
        """Create a BM25 database from all chunked documents."""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path

        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_documents(input_dir, output_file)
        print(f"BM25 database created at {output_file}")

    def parse_pdf_documents(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        if parallel:
            self.parse_pdf_documents_parallel(chunk_size=chunk_size, max_workers=max_workers)
        else:
            self.parse_pdf_documents_sequential()

    def parse_url_documents(self, urls: list[str] | list[dict], output_dir: Path = None, crawl_delay: float = 0.5):
        """Parse URL sources and save them in the same report JSON format used downstream."""
        target_dir = output_dir or self.paths.parsed_documents_path
        parser = URLParser(output_dir=target_dir, crawl_delay=crawl_delay)
        parser.parse_urls(urls)
        print(f"URL reports parsed and saved to {target_dir}")

    def parse_urls(self, root_path, urls_file, crawl_delay=0.5):
        """Parse URL pages and save them in pipeline-compatible JSON format."""

        urls_path = root_path / urls_file
        if not urls_path.exists():
            raise ValueError(f"URLs file not found: {urls_path}")

        suffix = urls_path.suffix.lower()
        with urls_path.open('r', encoding='utf-8') as file:
            if suffix == '.json':
                urls = json.load(file)
            elif suffix in {'.yaml', '.yml'}:
                urls = yaml.safe_load(file)
            else:
                raise ValueError(f"Unsupported URLs file format: {suffix}. Use .json, .yaml, or .yml")

        if not isinstance(urls, list):
            raise ValueError("URLs file must contain a list of URLs or URL objects")

        self.parse_url_documents(urls=urls, crawl_delay=crawl_delay)

        print(f"URLS parsed and saved")

    def process_parsed_documents(self):
         """Process parsed source documents through the pipeline:
         1. Merge/normalize to unified JSON structure
         2. Export to markdown
         3. Chunk the documents
         4. Create vector databases
         """
         print("Starting documents processing pipeline...")

         print("Step 1: Merging/normalizing documents...")
         self.merge_documents()

         print("Step 2: Exporting documents to markdown...")
         self.export_documents_to_markdown()

         print("Step 3: Chunking documents...")
         self.chunk_documents()

         print("Step 4: Creating vector databases...")
         self.create_vector_db()

         print("Documents processing pipeline completed successfully!")

    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        Returns the next available filename by adding a numbered suffix if the file exists.
        Example: If answers.json exists, returns answers_01.json, etc.
        """
        if not base_path.exists():
            return base_path

        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename

            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )

        output_path = self._get_next_available_filename(self.paths.answers_file_path)

        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    def evaluate_answers_similarity(self):
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )

        processing_result = processor.process_all_questions(
            output_path=self._get_next_available_filename(self.paths.answers_file_path),
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )

        questions = [item.get("q") for item in processor.questions]
        pipeline_answers = [item.get("value") for item in processing_result.get("questions", [])]
        true_answers = processing_result.get("true_answers", [])

        api_processor = APIProcessor(provider=self.run_config.api_provider)
        similarity_results = api_processor.get_answers_similarity(
            questions=questions,
            pipeline_answers=pipeline_answers,
            true_answers=true_answers,
            model=self.run_config.answering_model
        )

        scores = [item.get("score", 0.0) for item in similarity_results if item.get("score") is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"Average answer similarity score: {avg_score:.4f}")
        print("Per-question similarity:")
        for item in similarity_results:
            print(
                f"- Q: {item['question']}\n"
                f"  score: {item['score']}\n"
                f"  explanation: {item['explanation']}"
            )
        return similarity_results

    def evaluate_rag_quality(self):
        """Run one QA pass and evaluate both answer quality and retrieval quality."""
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )

        processing_result = processor.process_all_questions(
            output_path=self._get_next_available_filename(self.paths.answers_file_path),
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )

        questions = [item.get("q") for item in processor.questions]
        generated = processing_result.get("questions", [])
        pipeline_answers = [item.get("value") for item in generated]
        true_answers = processing_result.get("true_answers", [])
        retrieval_chunks = [item.get("retrieval_chunks", []) for item in generated]

        api_processor = APIProcessor(provider=self.run_config.api_provider)
        answer_similarity = api_processor.get_answers_similarity(
            questions=questions,
            pipeline_answers=pipeline_answers,
            true_answers=true_answers,
            model=self.run_config.answering_model
        )
        print("Similarity score counted")

        retrieval_quality = api_processor.calc_chunks_usefulness(
            questions=questions,
            true_answers=true_answers,
            retrieval_results=retrieval_chunks,
            model=self.run_config.answering_model
        )

        answer_scores = [item.get("score", 0.0) for item in answer_similarity if item.get("score") is not None]
        retrieval_usefulness_scores = [item.get("usefulness_scores") for item in retrieval_quality]
        retrieval_relevance_probabilitys = [item.get("topic_relevance_probabilitys") for item in retrieval_quality]

        retrieval_usefulness = [Pipeline.get_weighted_avg(i) for i in retrieval_usefulness_scores]
        retrieval_relevance = [Pipeline.get_weighted_avg(i) for i in retrieval_relevance_probabilitys]


        print(
            f"Average answer similarity score: {(sum(answer_scores) / len(answer_scores)) if answer_scores else 0.0:.4f}")
        print(
            f"Retrieval usefulness weighted_avg: {(sum(retrieval_usefulness) / len(retrieval_usefulness)) if retrieval_usefulness else 0.0:.4f}")
        print(
            f"Retrieval topic relevance weighted_avg: {(sum(retrieval_relevance) / len(retrieval_relevance)) if retrieval_relevance else 0.0:.4f}")

        return {
            "answer_similarity": answer_similarity,
            "retrieval_quality": retrieval_quality
        }

    @staticmethod
    def get_weighted_avg(scores: list[float]) -> float:
        scores = [score for score in scores if score is not None]
        if not scores:
            return 0
        s = sum([scores[i] / (math.log2(i + 2)) for i in range(len(scores))])
        n = sum([1 / (math.log2(i + 2)) for i in range(len(scores))])
        return s / n



preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),
                      "no_ser_tab": RunConfig(use_serialized_tables=False)}

base_config = RunConfig(
    parallel_requests=10,
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
    config_suffix="_base"
)

parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=20,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_pdr"
)

max_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=20,
    pipeline_details="Custom pdf parsing + table serialization + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_max"
)

max_no_ser_tab_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=20,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_max_no_ser_tab"
)

max_nst_o3m_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=25,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_nst_o3m"
)

max_st_o3m_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=25,
    pipeline_details="Custom pdf parsing + tables serialization + Router + vDB + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_st_o3m"
)

ibm_llama70b_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=10,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT + SO reparser; IBM WatsonX llm = llama-3.3-70b-instruct",
    api_provider="ibm",
    answering_model="meta-llama/llama-3-3-70b-instruct",
    config_suffix="_ibm_llama70b"
)

ibm_llama8b_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=10,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT + SO reparser; IBM WatsonX llm = llama-3.1-8b-instruct",
    api_provider="ibm",
    answering_model="meta-llama/llama-3-1-8b-instruct",
    config_suffix="_ibm_llama8b"
)

gemini_thinking_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    full_context=True,
    pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; llm = gemini-2.0-flash-thinking-exp-01-21",
    api_provider="gemini",
    answering_model="gemini-2.0-flash-thinking-exp-01-21",
    config_suffix="_gemini_thinking_fc"
)

gemini_flash_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    full_context=True,
    pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; llm = gemini-2.0-flash",
    api_provider="gemini",
    answering_model="gemini-2.0-flash",
    config_suffix="_gemini_flash_fc"
)

max_nst_o3m_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    llm_reranking_sample_size=36,
    top_n_retrieval=14,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini; top_n = 14; topn for rerank = 36",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_nst_o3m_bc"
)

ibm_llama70b_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    llm_reranking_sample_size=36,
    top_n_retrieval=14,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = llama-3.3-70b-instruct; top_n = 14; topn for rerank = 36",
    api_provider="ibm",
    answering_model="meta-llama/llama-3-3-70b-instruct",
    config_suffix="_ibm_llama70b_bc"
)

gemini_thinking_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    parallel_requests=1,
    top_n_retrieval=30,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = gemini-2.0-flash-thinking-exp-01-21; top_n = 30;",
    api_provider="gemini",
    answering_model="gemini-2.0-flash-thinking-exp-01-21",
    config_suffix="_gemini_thinking_bc"
)

configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config,
           "max_no_ser_tab": max_no_ser_tab_config,
           "max_nst_o3m": max_nst_o3m_config,  # This configuration returned the best results
           "max_st_o3m": max_st_o3m_config,
           "ibm_llama70b": ibm_llama70b_config,
           # This one won't work, because ibm api was avaliable only while contest was running
           "ibm_llama8b": ibm_llama8b_config,
           # This one won't work, because ibm api was avaliable only while contest was running
           "gemini_thinking": gemini_thinking_config}

# You can run any method right from this file with
# python .\src\pipeline.py
# Just uncomment the method you want to run
# You can also change the run_config to try out different configurations
if __name__ == "__main__":
    root_path = here() / "data" / "test_set"
    pipeline = Pipeline(root_path, run_config=max_nst_o3m_config)

    # This method parses pdf reports into a jsons. It creates jsons in the debug/data_01_parsed_reports. These jsons used in the next steps.
    # It also stores raw output of docling in debug/data_01_parsed_reports_debug, these jsons contain a LOT of metadata, and not used anywhere
    #pipeline.parse_pdf_documents_sequential()

    #pipeline.parse_urls(root_path, "links.yaml")

    # This method should be called only if you want run configs with serialized tables
    # It modifies the jsons in the debug/data_01_parsed_reports, adding a new field "serialized_table" to each table
    # pipeline.serialize_tables(max_workers=5)

    # This method converts jsons from the debug/data_01_parsed_reports into much simpler jsons, that is a list of pages in markdown
    # New jsons can be found in debug/data_02_merged_reports
    #pipeline.merge_documents()

    # This method exports the reports into plain markdown format. They used only for review and for full text search config: gemini_thinking_config
    # New files can be found in debug/data_03_reports_markdown
    #pipeline.export_documents_to_markdown()

    # This method splits the reports into chunks, that are used for vectorization
    # New jsons can be found in databases/chunked_reports
    #pipeline.chunk_documents()

    # This method creates vector databases from the chunked reports
    # New files can be found in databases/vector_dbs
    #pipeline.create_vector_db()

    # question = "Какой проходной балл на бюджет в 2025 году на программу «Программная инженерия» в Москве?"
    # retr = HybridRetriever(pipeline.paths.vector_db_dir)
    # retrieval_results  = retr.retrieve(question, llm_reranking_sample_size=pipeline.run_config.llm_reranking_sample_size, top_n=5)
    # for i in retrieval_results:
    #     print(i)
    #     print("*" * 80)

    # This method processes the questions and answers
    # Questions processing logic depends on the run_config
    # pipeline.process_questions()

    #pipeline.evaluate_answers_similarity()

    pipeline.evaluate_rag_quality()