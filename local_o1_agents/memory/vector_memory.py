import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
import threading
import time
from PIL import Image
import torchaudio

# Config
VECTOR_DIM = 384  # e.g., MiniLM
INDEX_FILE = 'vector_memory.index'
META_FILE = 'vector_memory_meta.json'
MAX_RAM_GB = 48
MAX_DISK_GB = 10
PRUNE_AGE_DAYS = 30
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def safe_faiss_add(index: Any, vectors: np.ndarray) -> None:
    """Helper function to add vectors to FAISS index with proper error handling."""
    try:
        index.add(vectors)  # type: ignore
    except Exception as e:
        print(f"Error adding vectors to FAISS index: {e}")

def safe_faiss_search(index: Any, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to search FAISS index with proper error handling."""
    try:
        return index.search(query_vectors, k)  # type: ignore
    except Exception as e:
        print(f"Error searching FAISS index: {e}")
        return np.array([[float('inf')]]), np.array([[-1]])

class VectorMemory:
    def __init__(self, dim=VECTOR_DIM, index_file=INDEX_FILE, meta_file=META_FILE):
        self.dim = dim
        self.index_file = index_file
        self.meta_file = meta_file
        self.lock = threading.Lock()
        self._load()
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL).to('mps' if torch.backends.mps.is_available() else 'cpu')
        self.cache_stats = {'hits': 0, 'misses': 0, 'retrieval_times': []}

    def _load(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
        if os.path.exists(self.meta_file):
            with open(self.meta_file) as f:
                self.meta = json.load(f)
        else:
            self.meta = []

    def _save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, 'w') as f:
            json.dump(self.meta, f)

    def embed(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(self.model.device)
            emb = self.model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        return emb.astype('float32')

    def add(self, prompt: str, output: str, meta: Optional[Dict[str, Any]] = None, sensitive: bool = False):
        if sensitive:
            return  # Do not persist sensitive info
        emb = self.embed(prompt)
        with self.lock:
            vectors = np.ascontiguousarray(emb.reshape(1, -1))
            safe_faiss_add(self.index, vectors)
            entry = {'prompt': prompt, 'output': output, 'meta': meta or {}, 'timestamp': time.time()}
            self.meta.append(entry)
            self._save()

    def add_image(self, image_path, output, meta=None, sensitive=False, agent=None):
        if sensitive:
            return
        if agent is None:
            from agents.agents import ImageAgent
            agent = ImageAgent()
        emb = agent.embed_image(image_path)
        if emb is None:
            return
        emb = np.array(emb).astype('float32')
        with self.lock:
            vectors = np.ascontiguousarray(emb.reshape(1, -1))
            safe_faiss_add(self.index, vectors)
            entry = {'image_path': image_path, 'output': output, 'meta': meta or {}, 'timestamp': time.time(), 'type': 'image'}
            self.meta.append(entry)
            self._save()

    def add_audio(self, audio_path, output, meta=None, sensitive=False, agent=None):
        if sensitive:
            return
        if agent is None:
            from agents.agents import AudioAgent
            agent = AudioAgent()
        # AudioAgent doesn't have embed_audio method, use transcribe instead
        transcription = agent.transcribe(audio_path)
        emb = self.embed(transcription)
        with self.lock:
            vectors = np.ascontiguousarray(emb.reshape(1, -1))
            safe_faiss_add(self.index, vectors)
            entry = {'audio_path': audio_path, 'output': output, 'meta': meta or {}, 'timestamp': time.time(), 'type': 'audio'}
            self.meta.append(entry)
            self._save()

    def retrieve(self, prompt: str, top_k=1, sim_threshold=0.85) -> Optional[str]:
        emb = self.embed(prompt)
        with self.lock:
            if self.index.ntotal == 0:
                self.cache_stats['misses'] += 1
                return None
            t0 = time.time()
            query_vectors = np.ascontiguousarray(emb.reshape(1, -1))
            distances, indices = safe_faiss_search(self.index, query_vectors, top_k)
            self.cache_stats['retrieval_times'].append(time.time() - t0)
            idx = indices[0][0]
            if distances[0][0] < (1 - sim_threshold):
                self.cache_stats['hits'] += 1
                return self.meta[idx]['output']
            else:
                self.cache_stats['misses'] += 1
                return None

    def retrieve_image(self, image_path, top_k=1, agent=None):
        if agent is None:
            from agents.agents import ImageAgent
            agent = ImageAgent()
        emb = agent.embed_image(image_path)
        with self.lock:
            if self.index.ntotal == 0:
                return None
            query_vectors = np.ascontiguousarray(emb.reshape(1, -1))
            distances, indices = safe_faiss_search(self.index, query_vectors, top_k)
            idx = indices[0][0]
            return self.meta[idx]['output'] if self.meta[idx]['type'] == 'image' else None

    def retrieve_audio(self, audio_path, top_k=1, agent=None):
        if agent is None:
            from agents.agents import AudioAgent
            agent = AudioAgent()
        # AudioAgent doesn't have embed_audio method, use transcribe instead
        transcription = agent.transcribe(audio_path)
        emb = self.embed(transcription)
        with self.lock:
            if self.index.ntotal == 0:
                return None
            query_vectors = np.ascontiguousarray(emb.reshape(1, -1))
            distances, indices = safe_faiss_search(self.index, query_vectors, top_k)
            idx = indices[0][0]
            return self.meta[idx]['output'] if self.meta[idx]['type'] == 'audio' else None

    def prune(self):
        # Remove entries older than PRUNE_AGE_DAYS or if disk usage exceeds limit
        now = time.time()
        keep = []
        for i, entry in enumerate(self.meta):
            if now - entry['timestamp'] < PRUNE_AGE_DAYS * 86400:
                keep.append((i, entry))
        if len(keep) < len(self.meta):
            idxs = np.array([i for i, _ in keep], dtype='int64')
            self.index = faiss.IndexFlatL2(self.dim)
            embeddings = np.vstack([self.embed(self.meta[i]['prompt']).reshape(1, -1) for i, _ in keep])
            vectors = np.ascontiguousarray(embeddings)
            safe_faiss_add(self.index, vectors)
            self.meta = [entry for _, entry in keep]
            self._save()
        # Disk usage check
        if os.path.getsize(self.index_file) / (1024**3) > MAX_DISK_GB:
            self.meta = self.meta[-10000:]  # Keep last 10k
            self.index = faiss.IndexFlatL2(self.dim)
            embeddings = np.vstack([self.embed(e['prompt']).reshape(1, -1) for e in self.meta])
            vectors = np.ascontiguousarray(embeddings)
            safe_faiss_add(self.index, vectors)
            self._save()

    def stats(self):
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'avg_retrieval_time_ms': 1000 * np.mean(self.cache_stats['retrieval_times']) if self.cache_stats['retrieval_times'] else 0,
            'entries': len(self.meta),
            'disk_usage_gb': os.path.getsize(self.index_file) / (1024**3) if os.path.exists(self.index_file) else 0
        }

    def export(self, out_file):
        with open(out_file, 'w') as f:
            json.dump(self.meta, f)

    def import_memory(self, in_file):
        with open(in_file) as f:
            entries = json.load(f)
        for entry in entries:
            self.add(entry['prompt'], entry['output'], entry.get('meta'), sensitive=False)

# Singleton instance
vector_memory = VectorMemory()
