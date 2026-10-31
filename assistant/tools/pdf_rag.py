"""Local PDF retrieval for the Paper Agent.

Uses TF-IDF (scikit-learn) so it works offline with no model downloads.
To upgrade to dense retrieval, replace `_build` / `search` with an embedding
model (e.g. sentence-transformers) and a vector store (FAISS / Chroma).
"""
import re
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


class PDFIndex:
    def __init__(self, chunk_size: int = 900, overlap: int = 150):
        self.chunk_size, self.overlap = chunk_size, overlap
        self.chunks: list[dict] = []
        self._vectorizer = None
        self._matrix = None

    @property
    def papers(self) -> list[str]:
        return sorted({c["source"] for c in self.chunks})

    def add_pdf(self, path: str | Path) -> int:
        path = Path(path)
        reader = PdfReader(str(path))
        added = 0
        for page_no, page in enumerate(reader.pages, 1):
            for piece in _chunk(page.extract_text() or "", self.chunk_size, self.overlap):
                self.chunks.append({"source": path.name, "page": page_no, "text": piece})
                added += 1
        self._vectorizer = self._matrix = None  # force rebuild
        return added

    def add_folder(self, folder: str | Path) -> int:
        total = 0
        for pdf in sorted(Path(folder).glob("**/*.pdf")):
            try:
                total += self.add_pdf(pdf)
            except Exception as e:
                print(f"Warning: could not read {pdf.name}: {e}")
        return total

    def _build(self):
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform([c["text"] for c in self.chunks])

    def search(self, query: str, k: int = 4) -> str:
        if not self.chunks:
            return "No papers are indexed. Put PDFs in the papers folder."
        if self._matrix is None:
            self._build()
        scores = linear_kernel(self._vectorizer.transform([query]), self._matrix).ravel()
        top = scores.argsort()[::-1][:k]
        hits = [(scores[i], self.chunks[i]) for i in top if scores[i] > 0]
        if not hits:
            return "No relevant passages found for that query."
        return "\n\n".join(
            f"[{c['source']}, p.{c['page']}] (score {s:.2f})\n{c['text']}" for s, c in hits
        )

    def list_papers(self, _: str = "") -> str:
        return "\n".join(self.papers) or "No papers indexed."
