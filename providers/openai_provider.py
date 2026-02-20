from typing import Optional, List, Dict, Any, Sequence
import numpy as np

class OpenAIEmbedder:
    """
    Uses OpenAI embeddings API (requires: pip install openai)
    Env var: OPENAI_API_KEY
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        batch_size: int = 64,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout_s = timeout_s

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI SDK not installed. Run: pip install openai"
            ) from e

        self._client = OpenAI()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from openai import APIError  # type: ignore

        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "input": batch,
            }
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions

            try:
                resp = self._client.embeddings.create(**kwargs)
            except APIError as e:
                raise RuntimeError(f"Embedding API error: {e}") from e

            vectors.extend([d.embedding for d in resp.data])

        arr = np.array(vectors, dtype=np.float32)
        # Normalize for cosine similarity
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr


class OpenAIChatLLM:
    """
    Uses OpenAI chat completions API (requires: pip install openai)
    Env var: OPENAI_API_KEY
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.2,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI SDK not installed. Run: pip install openai"
            ) from e

        self._client = OpenAI()

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
