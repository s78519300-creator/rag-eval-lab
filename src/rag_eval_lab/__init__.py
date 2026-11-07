"""rag-eval-lab: document Q&A with cited answers and a built-in evaluation harness."""

__version__ = "0.1.0"

from .evaluation import EvalExample, EvalReport, load_evalset, run_evaluation  # noqa: E402
from .generation import (  # noqa: E402
    Answer,
    ExtractiveGenerator,
    Generator,
    OpenAICompatibleGenerator,
)
from .ingest import Chunk, Document, chunk_document, chunk_documents, load_documents  # noqa: E402
from .pipeline import RAGPipeline  # noqa: E402
from .retrieval import BM25Index, RetrievalResult  # noqa: E402

__all__ = [
    "Answer",
    "BM25Index",
    "Chunk",
    "Document",
    "EvalExample",
    "EvalReport",
    "ExtractiveGenerator",
    "Generator",
    "OpenAICompatibleGenerator",
    "RAGPipeline",
    "RetrievalResult",
    "chunk_document",
    "chunk_documents",
    "load_documents",
    "load_evalset",
    "run_evaluation",
    "__version__",
]
