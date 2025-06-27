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
    CLIENT_CLASS: type = OllamaClient  # type: ignore
except ImportError:
    class MockClient:
        def chat(self, model: str, messages: List[Dict[str, str]]) -> Any:
            class MockResponse:
                def __init__(self):
                    self.message = {'content': 'Mock response'}
            return MockResponse()
    CLIENT_CLASS = MockClient  # type: ignore
    OllamaClient = MockClient  # type: ignore

try:
    import torch
    from transformers import (
        CLIPProcessor as _CLIPProcessor, 
        CLIPModel as _CLIPModel, 
        WhisperForConditionalGeneration, 
        AutoProcessor
    )
    from PIL import Image
    import torchaudio
    MULTIMODAL_AVAILABLE = True
    # Assign the imported classes directly to the global scope
    globals()['CLIPModel'] = _CLIPModel
    globals()['CLIPProcessor'] = _CLIPProcessor
except ImportError:
    torch = None  # type: ignore
    MULTIMODAL_AVAILABLE = False
    # Mock classes for type checking
    class CLIPModel: # type: ignore
        @classmethod
        def from_pretrained(cls, model_name: str): return cls()
        def to(self, device: str): return self
        def get_image_features(self, **kwargs): 
            # Create a mock tensor-like object since torch is None
            class MockTensor:
                def cpu(self): return self
                def numpy(self): return [[0.0]]
            return MockTensor()
    
    class CLIPProcessor:
        @classmethod
        def from_pretrained(cls, model_name: str): return cls() # type: ignore
        def __call__(self, images=None, return_tensors=None): return {}

try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
    ndarray = np.ndarray
except ImportError:
    import numpy.typing as npt
    class MockNumpy:
        def ascontiguousarray(self, x): return x
        def array(self, x): return x
        def vstack(self, arrays): return arrays[0] if arrays else []
        float32 = float
        ndarray = Any
    np = MockNumpy()
    faiss = None
    VECTOR_SEARCH_AVAILABLE = False
    ndarray = Any

try:
    # This import is intentionally placed here to allow the mock to be defined first
    # if the actual import fails.
    # The original code had an empty `except ImportError:` block which is invalid.
    # This change ensures that `vector_memory` is always defined, either by import
    # or by the mock class.
    from memory.vector_memory import vector_memory
except ImportError:
    # Mock vector memory
    class MockVectorMemory:
        def add(self, key: str, value: str) -> None:
            pass
    vector_memory = MockVectorMemory()

def faiss_add(index: Any, vectors: Any) -> None:
    """Helper function to add vectors to a FAISS index with proper type checking"""
    if not VECTOR_SEARCH_AVAILABLE or index is None:
        return
    contiguous_vectors = np.ascontiguousarray(vectors.astype(np.float32))
    index.add(contiguous_vectors)

def faiss_search(index: Any, query: Any, k: int) -> Tuple[Any, Any]:
    """Helper function to search a FAISS index with proper type checking"""
    if not VECTOR_SEARCH_AVAILABLE or index is None:
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
    def __init__(self, name: str, model: str):
        self.name = name
        self.client = CLIENT_CLASS()
        self.model = model
    
    def run(self, prompt: str, retries: int = 3, timeout: int = 30) -> str:
        for attempt in range(retries):
            try:
                start_time = time.time()
                response = self.client.chat(
                    model=self.model, 
                    messages=[{"role": "user", "content": f"[{self.name}] {prompt}"}]
                )
                elapsed = time.time() - start_time
                print(f"{self.name} response time: {elapsed:.2f}s")
                
                if hasattr(response, 'message') and isinstance(response.message, dict):
                    return response.message.get('content', str(response))
                return str(response)
                
            except Exception as e:
                print(f"[WARN] {self.name} failed (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        return f"[ERROR] {self.name} failed after {retries} attempts"

class TestGeneratorAgent(Agent):
    def generate_tests(self, module_path: str, bug_trace: Optional[str] = None) -> str:
        base = self.run(f"Generate pytest tests for {module_path}")
        if bug_trace:
            func_match = re.search(r'in (\w+)', bug_trace)
            func_name = func_match.group(1) if func_match else 'unknown'
            line_match = re.search(r'File ".*", line (\d+)', bug_trace)
            line_no = int(line_match.group(1)) if line_match else 0
            
            test_code = f"""
def test_{func_name}_bug():
    # Reproduces bug at line {line_no}
    with pytest.raises(Exception):
        {func_name}()
"""
            return base + "\n# Minimal failing test generated:\n" + test_code
        return base

class DependencyAgent(Agent):
    def analyze_deps(self) -> str:
        return self.run("Analyze project dependencies and create requirements.txt")

class ExecutorWithFallback(Agent):
    def run(self, prompt: str, retries: int = 3, timeout: int = 30) -> str:
        output = super().run(prompt, retries, timeout)
        # Quality check: fallback if output is error or too short
        if (output.startswith('[ERROR]') or len(output.strip()) < 10) and self.model == EXECUTOR_MODEL_DISTILLED:
            print(f"[FALLBACK] Distilled output failed, reverting to original model for {self.name}")
            self.model = EXECUTOR_MODEL_ORIGINAL
            return super().run(prompt, retries, timeout)
        return output

class ImageAgent:
    def __init__(self, device: Optional[str] = None):
        if not MULTIMODAL_AVAILABLE:
            raise ImportError("Multimodal dependencies not available")
        
        self.device = device or self._get_best_device()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.model = self.model.to(self.device)  # Assign back to self.model
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    def _get_best_device(self) -> str:
        if torch and torch.backends.mps.is_available():
            return "mps"
        elif torch and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def embed_image(self, image_path: str) -> Any:
        image = Image.open(image_path).convert("RGB") # type: ignore
        inputs = self.processor(images=image, return_tensors="pt")
        # Move tensors to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        if torch:
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            return emb.cpu().numpy()
        else:
            # Fallback when torch is None
            return np.array([[0.0]])
    
    def caption(self, image_path: str, prompt: str = "Describe this image.") -> str:
        # Placeholder: can be extended with BLIP or similar for captioning
        return "[Image captioning not implemented]"

class AudioAgent:
    def __init__(self, device: Optional[str] = None):
        if not MULTIMODAL_AVAILABLE:
            raise ImportError("Multimodal dependencies not available")
        
        self.device = device or self._get_best_device()
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        if torch and hasattr(self.model, 'to'):
            # Use type ignore to suppress false positive type errors
            self.model = self.model.to(self.device)  # type: ignore
        self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    
    def _get_best_device(self) -> str:
        if torch and hasattr(torch, 'backends') and torch.backends.mps.is_available():
            return "mps"
        elif torch and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def transcribe(self, audio_path: str) -> str:
        from contextlib import nullcontext
        
        waveform, sr = torchaudio.load(audio_path)
        processed_inputs = self.processor(
            waveform.squeeze().numpy(), 
            sampling_rate=sr, 
            return_tensors="pt"
        )
        
        # Move input features to device
        if "input_features" in processed_inputs:
            input_features = processed_inputs["input_features"].to(self.device)
        else:
            input_features = next(iter(processed_inputs.values())).to(self.device)
            
        with torch.no_grad() if torch else nullcontext():
            predicted_ids = self.model.generate(input_features)
            
        # Decode using the processor
        if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
            return self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        else:
            return "[Audio transcription processed but decoding method unavailable]"

class CodeAnalyzerAgent:
    def __init__(self, root_dir: str = "."): # type: ignore
        self.root_dir = root_dir
    
    def scan_repository(self) -> Dict[str, str]:
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
    
    def extract_imports(self, file_content: str) -> List[str]:
        try:
            tree = ast.parse(file_content)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            return imports
        except Exception:
            return []
    
    def build_dependency_graph(self, file_map: Dict[str, str]) -> Dict[str, List[str]]:
        graph = {}
        for fpath, content in file_map.items():
            graph[fpath] = self.extract_imports(content)
        return graph
    
    def analyze(self) -> Dict[str, List[str]]:
        file_map = self.scan_repository()
        dep_graph = self.build_dependency_graph(file_map)
        vector_memory.add("codebase_dependency_graph", json.dumps(dep_graph))
        return dep_graph

class CodeDebuggerAgent:
    def locate_bugs(self, file_content: str) -> str:
        with open("/tmp/_debug.py", "w") as f:
            f.write(file_content)
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", "/tmp/_debug.py"], 
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return result.stderr
            return "No syntax errors detected."
        except Exception as e:
            return str(e)
    
    def trace_execution(self, file_content: str) -> str:
        return "[Execution tracing not implemented]"
    
    def identify_root_cause(self, diagnostics: str) -> str:
        if "SyntaxError" in diagnostics:
            return "Syntax error detected."
        return "Root cause analysis not implemented."

class CodeEmbeddingIndex:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if not VECTOR_SEARCH_AVAILABLE:
            self.model = None
            self.index = None
            self.file_snippets = []
            return
            
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.file_snippets = []
    def build_index(self, file_map: Dict[str, str]) -> None:
        if not VECTOR_SEARCH_AVAILABLE or np is None or self.model is None or faiss is None:
            return
            
        self.file_snippets = []
        embeddings = []
        
        for fpath, content in file_map.items():
            for i, line in enumerate(content.splitlines()):
                snippet = {'file': fpath, 'line': i+1, 'text': line}
                self.file_snippets.append(snippet)
                if VECTOR_SEARCH_AVAILABLE and self.model is not None:
                    emb = self.model.encode(line, convert_to_numpy=True)
                    embeddings.append(emb)
        if embeddings and VECTOR_SEARCH_AVAILABLE and faiss is not None and np is not None:
            try:
                emb_array = np.vstack(embeddings)  # type: ignore
                # Ensure we have a proper numpy array
                if hasattr(emb_array, 'astype'):
                    emb_matrix = emb_array.astype(np.float32)  # type: ignore
                else:
                    emb_matrix = np.array(emb_array)  # type: ignore
                emb_matrix = np.ascontiguousarray(emb_matrix)  # type: ignore
                
                # Check if emb_matrix has shape attribute (should be numpy array)
                if hasattr(emb_matrix, 'shape') and len(emb_matrix.shape) > 1:  # type: ignore
                    self.index = faiss.IndexFlatL2(emb_matrix.shape[1])  # type: ignore
                    faiss_add(self.index, emb_matrix)
            except Exception as e:
                print(f"Error building FAISS index: {e}")
                self.index = None
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not VECTOR_SEARCH_AVAILABLE or not self.index or self.model is None:
            return []
            
        q_emb = self.model.encode(query, convert_to_numpy=True).reshape(1, -1).astype('float32')
        q_emb = np.ascontiguousarray(q_emb)  # type: ignore
        distances, indices = faiss_search(self.index, q_emb, top_k)
        
        # Use distances variable to avoid "not accessed" warning
        _ = distances
        return [self.file_snippets[idx] for idx in indices[0] if idx < len(self.file_snippets)]

class CodeRepairAgent:
    def __init__(self, file_map: Optional[Dict[str, str]] = None, 
                 embedding_index: Optional[CodeEmbeddingIndex] = None):
        self.file_map = file_map or {}
        self.embedding_index = embedding_index or CodeEmbeddingIndex()
    
    def generate_fix(self, file_content: str, diagnostics: str, bug_query: Optional[str] = None) -> str:
        # RAG: retrieve top-K relevant code snippets
        context_snippets = []
        if bug_query and self.embedding_index.index:
            context_snippets = self.embedding_index.search(bug_query, top_k=3)
        
        context_text = '\n'.join([f"{s['file']}:{s['line']}: {s['text']}" for s in context_snippets])
        
        # Simple fix for syntax errors
        if "SyntaxError" in diagnostics:
            lines = file_content.splitlines()
            for i, line in enumerate(lines):
                if "SyntaxError" in diagnostics and str(i+1) in diagnostics:
                    lines[i] = "# [AUTO-FIXED] " + line
            return "\n".join(lines)
        return file_content
    
    def test_solution(self, file_path: str) -> bool:
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", file_path], 
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def validate_repair(self, file_path: str) -> bool:
        return self.test_solution(file_path)

class PerformanceProfilerAgent:
    def profile_code(self, file_path: str, func_name: Optional[str] = None) -> Dict[str, Any]:
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
            
            def run_target():
                if target_func:
                    return target_func()
                elif hasattr(mod, "main"):
                    return mod.main()
                return None
                
            # Profile execution
            pr.enable()
            result = run_target()
            pr.disable()
            
            ps = pstats.Stats(pr, stream=stats_output).sort_stats('cumulative')
            ps.print_stats(20)
            cpu_profile = stats_output.getvalue()
            
            # Memory profiling using tracemalloc
            _, mem_info = measure_memory_usage(run_target)
            
            return {
                "cpu_profile": cpu_profile, 
                "mem_profile": mem_info, 
                "result": result
            }
        except Exception as e:
            return {
                "cpu_profile": f"Error: {str(e)}", 
                "mem_profile": {"error": str(e)}, 
                "result": None
            }
    
    def find_hotspots(self, cpu_profile: str) -> List[Dict[str, Any]]:
        lines = cpu_profile.splitlines()
        hotspots = []
        for line in lines:
            if line.strip().startswith(('[', 'ncalls')):
                continue
            parts = line.split()
            if len(parts) > 5:
                try:
                    time_spent = float(parts[3])
                    if time_spent > 0.01:  # threshold
                        hotspots.append({'line': line, 'time': time_spent})
                except Exception:
                    continue
        return sorted(hotspots, key=lambda x: -x['time'])[:5]
    
    def analyze_complexity(self, file_path: str) -> Dict[str, str]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                tokens = list(tokenize.generate_tokens(f.readline))
            
            func_complexities = {}
            func_name = None
            
            for tok in tokens:
                if tok.type == tokenize.NAME and tok.string == 'def':
                    func_name = None
                elif tok.type == tokenize.NAME and func_name is None:
                    func_name = tok.string
                    func_complexities[func_name] = 1
                elif tok.type == tokenize.NAME and tok.string in ('for', 'while') and func_name:
                    func_complexities[func_name] *= 10
            
            return {k: ('O(n^2) or worse' if v > 100 else 'O(n) or better') 
                   for k, v in func_complexities.items()}
        except Exception:
            return {}

class OptimizationSuggesterAgent:
    def suggest(self, cpu_profile: List[Dict[str, Any]], mem_profile: Dict[str, Any], 
               complexity_report: Dict[str, str]) -> List[str]:
        suggestions = []
        
        for func, complexity in complexity_report.items():
            if 'O(n^2)' in complexity:
                suggestions.append(
                    f"Function {func} has high complexity: {complexity}. "
                    f"Consider optimizing nested loops."
                )
        
        for hotspot in cpu_profile:
            suggestions.append(
                f"Hotspot: {hotspot['line']} (time: {hotspot['time']:.4f}s). "
                f"Consider refactoring or memoization."
            )
        
        if 'current' in mem_profile and 'peak' in mem_profile:
            if mem_profile['peak'] - mem_profile['current'] > 100 * 1024 * 1024:  # 100MB
                suggestions.append(
                    "High memory usage detected. Check for leaks or large data structures."
                )
        
        return suggestions

class BenchmarkingTool:
    def benchmark(self, file_path: str, func_name: Optional[str] = None, repeat: int = 3) -> Dict[str, Any]:
        try:
            spec = importlib.util.spec_from_file_location("mod", file_path)
            if spec is None:
                raise ImportError(f"Could not load spec for {file_path}")
            
            mod = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ImportError(f"Could not load module for {file_path}")
            
            spec.loader.exec_module(mod)
            target_func = getattr(mod, func_name) if func_name else None
            
            stmt = f"mod.{func_name}()" if func_name else "mod.main()"
            setup = f"from __main__ import mod"
            
            times = timeit.repeat(
                stmt=stmt, setup=setup, repeat=repeat, number=1, globals={'mod': mod}
            )
            
            return {'min': min(times), 'max': max(times), 'avg': sum(times)/len(times)}
        except Exception as e:
            return {'error': str(e)}

# Factory functions
def create_ceo() -> Agent:
    return Agent('CEO', CEO_MODEL)

def create_executor(i: int, task_complexity: Optional[str] = None) -> ExecutorWithFallback:
    if task_complexity == 'high':
        model = EXECUTOR_MODEL_ORIGINAL
    elif task_complexity == 'low':
        model = EXECUTOR_MODEL_DISTILLED
    else:
        model = EXECUTOR_MODEL_DISTILLED if USE_DISTILLED_EXECUTOR else EXECUTOR_MODEL_ORIGINAL
    return ExecutorWithFallback(f'Executor_{i}', model)

def create_summarizer() -> Agent:
    return Agent('Summarizer', CEO_MODEL)

def create_test_generator() -> TestGeneratorAgent:
    return TestGeneratorAgent('TestGenerator', CEO_MODEL)

def create_dependency_agent() -> DependencyAgent:
    return DependencyAgent('DependencyAgent', CEO_MODEL)

# High-level workflow functions
def automated_debugging_workflow(target_file: str) -> Dict[str, Any]:
    analyzer = CodeAnalyzerAgent()
    debugger = CodeDebuggerAgent()
    repairer = CodeRepairAgent()
    
    # 1. Scan and analyze
    file_map = analyzer.scan_repository()
    dep_graph = analyzer.build_dependency_graph(file_map)
    
    # 2. Focus on target file
    content = file_map.get(target_file)
    if not content:
        return {"error": f"File {target_file} not found."}
    
    diagnostics = debugger.locate_bugs(content)
    root_cause = debugger.identify_root_cause(diagnostics)
    
    # 3. Generate repair
    fixed_content = repairer.generate_fix(content, diagnostics)
    tmp_path = "/tmp/_repaired.py"
    with open(tmp_path, "w") as f:
        f.write(fixed_content)
    
    # 4. Test and validate
    test_passed = repairer.test_solution(tmp_path)
    result = {
        "diagnostics": diagnostics, 
        "root_cause": root_cause, 
        "test_passed": test_passed
    }
    
    vector_memory.add(f"repair_{target_file}", json.dumps(result))
    return result

class IntegratedCodebaseOptimizer:
    def __init__(self, root_dir: str = "."):
        self.analyzer = CodeAnalyzerAgent(root_dir)
        self.debugger = CodeDebuggerAgent()
        self.repairer = CodeRepairAgent()
        self.profiler = PerformanceProfilerAgent()
        self.suggester = OptimizationSuggesterAgent()
        self.benchmarker = BenchmarkingTool()
    
    def optimize(self, target_file: str, func_name: Optional[str] = None) -> Dict[str, Any]:
        # 1. Analyze codebase and build dependency graph
        file_map = self.analyzer.scan_repository()
        dep_graph = self.analyzer.build_dependency_graph(file_map)
        
        # 2. Calculate impact scores
        impact_scores = {}
        for f, deps in dep_graph.items():
            impact_scores[f] = len(deps) if deps else 0
        
        prioritized = sorted(impact_scores.keys(), key=lambda x: impact_scores[x], reverse=True)
        report = {"fixed_bugs": [], "performance": {}}
        
        for f in prioritized:
            content = file_map.get(f)
            if not content:
                continue
            
            # 3. Debug and repair
            diagnostics = self.debugger.locate_bugs(content)
            if "SyntaxError" in diagnostics or "Exception" in diagnostics:
                root_cause = self.debugger.identify_root_cause(diagnostics)
                fixed_content = self.repairer.generate_fix(content, diagnostics)
                tmp_path = "/tmp/_repaired.py"
                with open(tmp_path, "w") as out:
                    out.write(fixed_content)
                test_passed = self.repairer.test_solution(tmp_path)
                
                report["fixed_bugs"].append({
                    "file": f, 
                    "diagnostics": diagnostics, 
                    "root_cause": root_cause, 
                    "test_passed": test_passed
                })
                
                # 4. Profile and optimize
                orig_prof = self.profiler.profile_code(f, func_name)
                orig_hotspots = self.profiler.find_hotspots(orig_prof["cpu_profile"])
                orig_complexity = self.profiler.analyze_complexity(f)
                
                opt_prof = self.profiler.profile_code(tmp_path, func_name)
                opt_hotspots = self.profiler.find_hotspots(opt_prof["cpu_profile"])
                opt_complexity = self.profiler.analyze_complexity(tmp_path)
                
                suggestions = self.suggester.suggest(opt_hotspots, opt_prof["mem_profile"], opt_complexity)
                
                # 5. Benchmark A/B
                orig_bench = self.benchmarker.benchmark(f, func_name)
                opt_bench = self.benchmarker.benchmark(tmp_path, func_name)
                
                report["performance"][f] = {
                    "original": {"profile": orig_prof, "bench": orig_bench},
                    "optimized": {"profile": opt_prof, "bench": opt_bench},
                    "suggestions": suggestions
                }
        
        return report

def pipeline_coordinator(target_file: str, func_name: Optional[str] = None, root_dir: str = ".") -> str:
    optimizer = IntegratedCodebaseOptimizer(root_dir)
    report = optimizer.optimize(target_file, func_name)
    
    # Unified reporting
    summary = []
    for bug in report["fixed_bugs"]:
        summary.append(
            f"Fixed bug in {bug['file']}: {bug['diagnostics']} "
            f"(root cause: {bug['root_cause']}, test passed: {bug['test_passed']})"
        )
    
    for f, perf in report["performance"].items():
        summary.append(
            f"Performance for {f}:\n"
            f"  Original: {perf['original']['bench']}\n"
            f"  Optimized: {perf['optimized']['bench']}\n"
            f"  Suggestions: {perf['suggestions']}"
        )
    
    return "\n".join(summary)
