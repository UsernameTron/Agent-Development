import sys
import os
import time
import json
import ast
import re
import io
import tokenize
import cProfile
import pstats
import importlib
import importlib.util
import subprocess
import timeit
import tracemalloc
from typing import Dict, List, Optional, Any, Tuple, Union

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import config with fallback
try:
    from config.config import (
        CEO_MODEL, FAST_MODEL, EXECUTOR_MODEL_ORIGINAL, 
        EXECUTOR_MODEL_DISTILLED, USE_DISTILLED_EXECUTOR
    )
except ImportError:
    # Fallback configuration
    CEO_MODEL = 'phi3.5'
    FAST_MODEL = 'phi3.5'
    EXECUTOR_MODEL_ORIGINAL = 'phi3.5'
    EXECUTOR_MODEL_DISTILLED = 'executor-distilled'
    USE_DISTILLED_EXECUTOR = True

# Third-party imports with error handling
try:
    from ollama import Client as OllamaClient
    Client = OllamaClient
except ImportError:
    class Client:
        def chat(self, model: str, messages: List[Dict[str, str]]) -> Any:
            class MockResponse:
                def __init__(self):
                    self.message = {'content': 'Mock response'}
            return MockResponse()

try:
    import torch
    from transformers import (
        CLIPProcessor, CLIPModel, WhisperProcessor, 
        WhisperForConditionalGeneration, AutoProcessor
    )
    from PIL import Image
    import torchaudio
    MULTIMODAL_AVAILABLE = True
except ImportError:
    torch = None
    MULTIMODAL_AVAILABLE = False

try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    np = None
    faiss = None
    VECTOR_SEARCH_AVAILABLE = False

try:
    from memory.vector_memory import vector_memory
except ImportError:
    # Mock vector memory
    class MockVectorMemory:
        def add(self, key: str, value: str) -> None:
            pass
    vector_memory = MockVectorMemory()

# Helper functions for FAISS operations with proper type handling
def faiss_add(index: Any, vectors: Any) -> None:
    """Helper function to add vectors to a FAISS index with proper type checking"""
    if not VECTOR_SEARCH_AVAILABLE or index is None or np is None:
        return
    contiguous_vectors = np.ascontiguousarray(vectors.astype(np.float32))
    index.add(contiguous_vectors)

def faiss_search(index: Any, query: Any, k: int) -> Tuple[Any, Any]:
    """Helper function to search a FAISS index with proper type checking"""
    if not VECTOR_SEARCH_AVAILABLE or index is None or np is None:
        return np.array([]), np.array([])
    contiguous_query = np.ascontiguousarray(query.astype(np.float32))
    distances, indices = index.search(contiguous_query, k)
    return distances, indices

def measure_memory_usage(func):
    """Memory profiling using tracemalloc instead of memory_profiler"""
    tracemalloc.start()
    try:
        result = func()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, {'current': current, 'peak': peak}
    except Exception as e:
        tracemalloc.stop()
        return None, {'error': str(e)}

class Agent:
    def __init__(self, name, model):
        self.name = name
        self.client = Client()
        self.model = model
    
    def run(self, prompt, retries=3, timeout=30):
        for attempt in range(retries):
            try:
                start_time = time.time()
                response = self.client.chat(model=self.model, messages=[{"role": "user", "content": f"[{self.name}] {prompt}"}])
                elapsed = time.time() - start_time
                print(f"{self.name} response time: {elapsed:.2f}s")
                return response.message['content'] if hasattr(response, 'message') and 'content' in response.message else str(response)
            except Exception as e:
                print(f"[WARN] {self.name} failed (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return f"[ERROR] {self.name} failed after {retries} attempts"

class TestGeneratorAgent(Agent):
    def generate_tests(self, module_path, bug_trace=None):
        base = self.run(f"Generate pytest tests for {module_path}")
        if bug_trace:
            func_match = re.search(r'in (\w+)', bug_trace)
            func_name = func_match.group(1) if func_match else None
            line_match = re.search(r'File ".*", line (\d+)', bug_trace)
            line_no = int(line_match.group(1)) if line_match else None
            test_code = f"def test_{func_name or 'bug'}():\n    # Reproduces bug at line {line_no}\n    with pytest.raises(Exception):\n        {func_name or 'function'}()\n"
            return base + "\n# Minimal failing test generated:\n" + test_code
        return base

class DependencyAgent(Agent):
    def analyze_deps(self):
        return self.run("Analyze project dependencies and create requirements.txt")

def create_ceo(): 
    return Agent('CEO', CEO_MODEL)

def create_executor(i, task_complexity=None):
    if task_complexity == 'high':
        model = EXECUTOR_MODEL_ORIGINAL
    elif task_complexity == 'low':
        model = EXECUTOR_MODEL_DISTILLED
    else:
        model = EXECUTOR_MODEL_DISTILLED if USE_DISTILLED_EXECUTOR else EXECUTOR_MODEL_ORIGINAL
    return ExecutorWithFallback(f'Executor_{i}', model)

class ExecutorWithFallback(Agent):
    def run(self, prompt, retries=3, timeout=30):
        output = super().run(prompt, retries, timeout)
        if (output.startswith('[ERROR]') or len(output.strip()) < 10) and self.model == EXECUTOR_MODEL_DISTILLED:
            print(f"[FALLBACK] Distilled output failed, reverting to original model for {self.name}")
            self.model = EXECUTOR_MODEL_ORIGINAL
            return super().run(prompt, retries, timeout)
        return output

class ImageAgent:
    def __init__(self, device=None):
        if not MULTIMODAL_AVAILABLE:
            raise ImportError("Multimodal dependencies not available")
        
        self.device = device or self._get_best_device()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.model = self.model.to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    def _get_best_device(self):
        if torch and hasattr(torch, 'backends') and torch.backends.mps.is_available():
            return "mps"
        elif torch and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb.cpu().numpy().astype('float32')
    
    def caption(self, image_path, prompt="Describe this image."):
        return "[Image captioning not implemented]"

class AudioAgent:
    def __init__(self, device=None):
        if not MULTIMODAL_AVAILABLE:
            raise ImportError("Multimodal dependencies not available")
        
        self.device = device or self._get_best_device()
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    
    def _get_best_device(self):
        if torch and hasattr(torch, 'backends') and torch.backends.mps.is_available():
            return "mps"
        elif torch and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def transcribe(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        processed_inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
        
        if "input_features" in processed_inputs:
            input_features = processed_inputs["input_features"].to(self.device)
        else:
            input_features = next(iter(processed_inputs.values())).to(self.device)
            
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
            
        if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
            return self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        else:
            return "[Audio transcription processed but decoding method unavailable]"

class CodeAnalyzerAgent:
    def __init__(self, root_dir="."):
        self.root_dir = root_dir
    
    def scan_repository(self):
        file_map = {}
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.go')):
                    fpath = os.path.join(dirpath, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        file_map[fpath] = content
                    except Exception as e:
                        print(f"Error reading {fpath}: {e}")
        return file_map
    
    def extract_imports(self, file_content):
        try:
            tree = ast.parse(file_content)
            imports = []
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    for alias in n.names:
                        imports.append(alias.name)
                elif isinstance(n, ast.ImportFrom) and n.module:
                    imports.append(n.module)
            return imports
        except Exception:
            return []
    
    def build_dependency_graph(self, file_map):
        graph = {}
        for fpath, content in file_map.items():
            graph[fpath] = self.extract_imports(content)
        return graph
    
    def analyze(self):
        file_map = self.scan_repository()
        dep_graph = self.build_dependency_graph(file_map)
        vector_memory.add("codebase_dependency_graph", json.dumps(dep_graph))
        return dep_graph

class CodeEmbeddingIndex:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not VECTOR_SEARCH_AVAILABLE:
            self.model = None
            self.index = None
            self.file_snippets = []
            return
        
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.file_snippets = []
    
    def build_index(self, file_map):
        if not VECTOR_SEARCH_AVAILABLE or not self.model:
            return
        
        self.file_snippets = []
        embeddings = []
        for fpath, content in file_map.items():
            for i, line in enumerate(content.splitlines()):
                snippet = {'file': fpath, 'line': i+1, 'text': line}
                self.file_snippets.append(snippet)
                embeddings.append(self.model.encode(line, convert_to_numpy=True))
        
        if embeddings and np and faiss:
            emb_matrix = np.vstack(embeddings).astype('float32')
            emb_matrix = np.ascontiguousarray(emb_matrix)
            self.index = faiss.IndexFlatL2(emb_matrix.shape[1])
            faiss_add(self.index, emb_matrix)
    
    def search(self, query, top_k=5):
        if not VECTOR_SEARCH_AVAILABLE or not self.index or not np or not self.model:
            return []
        
        q_emb = self.model.encode(query, convert_to_numpy=True).reshape(1, -1).astype('float32')
        q_emb = np.ascontiguousarray(q_emb)
        distances, indices = faiss_search(self.index, q_emb, top_k)
        return [self.file_snippets[idx] for idx in indices[0] if idx < len(self.file_snippets)]

class PerformanceProfilerAgent:
    def __init__(self):
        pass
    
    def profile_code(self, file_path, func_name=None):
        pr = cProfile.Profile()
        stats_output = io.StringIO()
        
        try:
            spec = importlib.util.spec_from_file_location("mod", file_path)
            if spec is None:
                raise ImportError(f"Could not load spec for {file_path}")
            
            mod = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ImportError(f"Could not load module for {file_path}")
            
            spec.loader.exec_module(mod)
            target_func = getattr(mod, func_name) if func_name and hasattr(mod, func_name) else None
            
            def run():
                if target_func:
                    return target_func()
                elif hasattr(mod, "main"):
                    return mod.main()
                return None
                
            pr.enable()
            result = run()
            pr.disable()
            
            ps = pstats.Stats(pr, stream=stats_output).sort_stats('cumulative')
            ps.print_stats(20)
            cpu_profile = stats_output.getvalue()
            
            _, mem_info = measure_memory_usage(run)
            
            return {"cpu_profile": cpu_profile, "mem_profile": mem_info, "result": result}
        except Exception as e:
            return {"cpu_profile": f"Error: {str(e)}", "mem_profile": {"error": str(e)}, "result": None}

def create_summarizer(): 
    return Agent('Summarizer', CEO_MODEL)

def create_test_generator(): 
    return TestGeneratorAgent('TestGenerator', CEO_MODEL)

def create_dependency_agent(): 
    return DependencyAgent('DependencyAgent', CEO_MODEL)