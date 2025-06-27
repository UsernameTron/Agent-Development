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

# Import the new self-aware agent system
try:
    from .self_aware_agent import SelfAwareAgent
    SELF_AWARE_AVAILABLE = True
except ImportError:
    print("SelfAwareAgent not available, using basic Agent")
    SELF_AWARE_AVAILABLE = False

# Third-party imports with error handling
try:
    from ollama import Client as OllamaClient
    Client: type = OllamaClient # type: ignore
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
        return [], []
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
    """Enhanced Agent class with optional self-awareness capabilities"""
    
    def __init__(self, name, model, specialization=None):
        if SELF_AWARE_AVAILABLE:
            # Initialize as SelfAwareAgent if available
            self._initialize_as_self_aware(name, model, specialization)
        else:
            # Fallback to basic agent
            self._initialize_as_basic(name, model)
    
    def _initialize_as_self_aware(self, name, model, specialization):
        """Initialize with SelfAwareAgent capabilities"""
        # Create a SelfAwareAgent instance and copy its attributes
        self._self_aware_agent = SelfAwareAgent(name, model, specialization)
        
        # Copy attributes for compatibility
        self.name = self._self_aware_agent.name
        self.model = self._self_aware_agent.model
        self.client = self._self_aware_agent.client
        self.specialization = specialization
        
        # Expose self-aware capabilities
        self.agent_id = self._self_aware_agent.agent_id
        self.performance_history = self._self_aware_agent.performance_history
        self.knowledge_base = self._self_aware_agent.knowledge_base
        self.improvement_tracker = self._self_aware_agent.improvement_tracker
        self.dna = self._self_aware_agent.dna
        self.current_metrics = self._self_aware_agent.current_metrics
        self.task_counter = self._self_aware_agent.task_counter
    
    def _initialize_as_basic(self, name, model):
        """Initialize as basic agent (fallback)"""
        self.name = name
        self.client = Client()
        self.model = model
        self.specialization = None
        
        # Stub attributes for compatibility
        self.agent_id = f"basic_{name}_{int(time.time())}"
        self.task_counter = 0
    
    def run(self, prompt, retries=3, timeout=30):
        """Enhanced run method with optional self-awareness"""
        if SELF_AWARE_AVAILABLE and hasattr(self, '_self_aware_agent'):
            # Use SelfAwareAgent's enhanced run method
            return self._self_aware_agent.run(prompt, retries, timeout)
        else:
            # Fallback to basic implementation
            return self._basic_run(prompt, retries, timeout)
    
    def _basic_run(self, prompt, retries=3, timeout=30):
        """Basic run implementation (original logic)"""
        self.task_counter += 1
        
        for attempt in range(retries):
            try:
                start_time = time.time()
                # Remove unsupported timeout argument
                response = self.client.chat(model=self.model, messages=[{"role": "user", "content": f"[{self.name}] {prompt}"}])
                elapsed = time.time() - start_time
                print(f"{self.name} response time: {elapsed:.2f}s")
                return response.message['content'] if hasattr(response, 'message') and 'content' in response.message else str(response)
            except Exception as e:
                print(f"[WARN] {self.name} failed (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        return f"[ERROR] {self.name} failed after {retries} attempts"
    
    # Self-awareness methods (available when SELF_AWARE_AVAILABLE is True)
    def analyze_self(self):
        """Get self-analysis report"""
        if SELF_AWARE_AVAILABLE and hasattr(self, '_self_aware_agent'):
            return self._self_aware_agent.analyze_self()
        else:
            return {"error": "Self-awareness not available"}
    
    def improve_self(self, improvement_plan):
        """Apply self-improvement plan"""
        if SELF_AWARE_AVAILABLE and hasattr(self, '_self_aware_agent'):
            return self._self_aware_agent.improve_self(improvement_plan)
        else:
            return False
    
    def teach_knowledge(self, target_agent, knowledge_type):
        """Teach knowledge to another agent"""
        if SELF_AWARE_AVAILABLE and hasattr(self, '_self_aware_agent'):
            return self._self_aware_agent.teach_knowledge(target_agent, knowledge_type)
        else:
            return False
    
    def receive_knowledge(self, knowledge_package):
        """Receive knowledge from another agent"""
        if SELF_AWARE_AVAILABLE and hasattr(self, '_self_aware_agent'):
            return self._self_aware_agent.receive_knowledge(knowledge_package)
        else:
            return False
    
    def get_performance_summary(self):
        """Get performance summary"""
        if SELF_AWARE_AVAILABLE and hasattr(self, '_self_aware_agent'):
            return self._self_aware_agent.get_performance_summary()
        else:
            return {
                'agent_id': self.agent_id,
                'name': self.name,
                'task_count': self.task_counter,
                'self_aware': False
            }

class TestGeneratorAgent(Agent):
    """Test generator agent with enhanced capabilities"""
    
    def __init__(self, name, model, specialization='testing'):
        super().__init__(name, model, specialization)
    
    def generate_tests(self, module_path, bug_trace=None):
        base = self.run(f"Generate pytest tests for {module_path}")
        if bug_trace:
            # Extract function name and error line from bug_trace
            func_match = re.search(r'in (\w+)', bug_trace)
            func_name = func_match.group(1) if func_match else None
            line_match = re.search(r'File ".*", line (\d+)', bug_trace)
            line_no = int(line_match.group(1)) if line_match else None
            # Minimal failing test
            test_code = f"def test_{func_name or 'bug'}():\n    # Reproduces bug at line {line_no}\n    with pytest.raises(Exception):\n        {func_name or 'function'}()\n"
            return base + "\n# Minimal failing test generated:\n" + test_code
        return base

class DependencyAgent(Agent):
    """Dependency analysis agent with enhanced capabilities"""
    
    def __init__(self, name, model, specialization='dependency_analysis'):
        super().__init__(name, model, specialization)
    
    def analyze_deps(self):
        return self.run("Analyze project dependencies and create requirements.txt")

def create_ceo(specialization='strategic_planning'): 
    """Create CEO agent with strategic planning specialization"""
    return Agent('CEO', CEO_MODEL, specialization)

def create_executor(i, task_complexity=None, specialization='task_execution'):
    """Create executor agent with enhanced capabilities"""
    # Automatic model selection based on task complexity
    if task_complexity == 'high':
        model = EXECUTOR_MODEL_ORIGINAL
    elif task_complexity == 'low':
        model = EXECUTOR_MODEL_DISTILLED
    else:
        model = EXECUTOR_MODEL_DISTILLED if USE_DISTILLED_EXECUTOR else EXECUTOR_MODEL_ORIGINAL
    
    # Create with specialization
    executor = ExecutorWithFallback(f'Executor_{i}', model, specialization)
    return executor

class ExecutorWithFallback(Agent):
    """Executor agent with fallback capability and enhanced self-awareness"""
    
    def __init__(self, name, model, specialization='task_execution'):
        super().__init__(name, model, specialization)
        self.original_model = EXECUTOR_MODEL_ORIGINAL
        self.distilled_model = EXECUTOR_MODEL_DISTILLED
    
    def run(self, prompt, retries=3, timeout=30):
        output = super().run(prompt, retries, timeout)
        # Quality check: fallback if output is error or too short
        if (output.startswith('[ERROR]') or len(output.strip()) < 10) and self.model == self.distilled_model:
            print(f"[FALLBACK] Distilled output failed, reverting to original model for {self.name}")
            self.model = self.original_model
            
            # Update the underlying model if using self-aware agent
            if SELF_AWARE_AVAILABLE and hasattr(self, '_self_aware_agent'):
                self._self_aware_agent.model = self.original_model
            
            return super().run(prompt, retries, timeout)
        return output

class ImageAgent:
    def __init__(self, device=None):
        if not MULTIMODAL_AVAILABLE:
            raise ImportError("Multimodal dependencies not available")
        
        self.device = device or self._get_best_device()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        if torch is not None and hasattr(self.model, 'to'):
            self.model = self.model.to(self.device) # type: ignore
        processor_result = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        # Handle case where from_pretrained returns a tuple
        if isinstance(processor_result, tuple):
            self.processor = processor_result[0]
        else:
            self.processor = processor_result
    
    def _get_best_device(self):
        if torch and hasattr(torch, 'backends') and torch.backends.mps.is_available():
            return "mps"
        elif torch and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def embed_image(self, image_path):
        if not MULTIMODAL_AVAILABLE or torch is None:
            raise ImportError("Multimodal dependencies not available")
            
        image = Image.open(image_path).convert("RGB") # type: ignore
        inputs = self.processor(images=image, return_tensors="pt")
        # Move tensors to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb.cpu().numpy().astype('float32')
    
    def caption(self, image_path, prompt="Describe this image."):
        # Placeholder: can be extended with BLIP or similar for captioning
        return "[Image captioning not implemented]"

class AudioAgent:
    def __init__(self, device=None):
        if not MULTIMODAL_AVAILABLE:
            raise ImportError("Multimodal dependencies not available")
        
        self.device = device or self._get_best_device()
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        if torch is not None and hasattr(self.model, 'to'): # type: ignore
            self.model = self.model.to(self.device) # type: ignore
        # Use AutoProcessor instead to ensure we get the correct processor type
        self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    
    def _get_best_device(self):
        if torch and hasattr(torch, 'backends') and torch.backends.mps.is_available():
            return "mps"
        elif torch and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def transcribe(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        # Process the audio input
        processed_inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
        # Move input features to the device
        if "input_features" in processed_inputs:
            input_features = processed_inputs["input_features"].to(self.device)
        else:
            # Fallback if the keys are different
            input_features = next(iter(processed_inputs.values())).to(self.device)

        # Use torch.no_grad() if torch is available, otherwise use a null context
        from contextlib import nullcontext
        with torch.no_grad() if torch else nullcontext():
            predicted_ids = self.model.generate(input_features)
            
        # For decoding, we use the tokenizer part of the processor if available
        if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
            return self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        else:
            # Simple fallback - return a placeholder
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
                    try:  # Add error handling for file reading
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

class CodeDebuggerAgent:
    def __init__(self):
        pass
    
    def locate_bugs(self, file_content):
        # Placeholder: Use StaticAnalysisTool or linting
        import subprocess
        with open("/tmp/_debug.py", "w") as f:
            f.write(file_content)
        try:
            result = subprocess.run(["python3", "-m", "py_compile", "/tmp/_debug.py"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return result.stderr
            return "No syntax errors detected."
        except Exception as e:
            return str(e)
    
    def trace_execution(self, file_content):
        # Placeholder: Could use sys.settrace or coverage
        return "[Execution tracing not implemented]"
    
    def identify_root_cause(self, diagnostics):
        # Placeholder: Use heuristics or LLM
        if "SyntaxError" in diagnostics:
            return "Syntax error detected."
        return "Root cause analysis not implemented."

class CodeRepairAgent:
    def __init__(self, file_map=None, embedding_index=None):
        self.file_map = file_map or {}
        self.embedding_index = embedding_index or CodeEmbeddingIndex()
    
    def generate_fix(self, file_content, diagnostics, bug_query=None):
        # RAG: retrieve top-K relevant code snippets
        context_snippets = []
        if bug_query and self.embedding_index.index:
            context_snippets = self.embedding_index.search(bug_query, top_k=3)
        context_text = '\n'.join([f"{s['file']}:{s['line']}: {s['text']}" for s in context_snippets])
        # Placeholder: Use LLM or pattern-based fix, with context
        fix_prompt = f"Bug: {diagnostics}\nContext:\n{context_text}\nCode:\n{file_content}\nSuggest a fix."
        # Here, you would call an LLM with fix_prompt; fallback to naive fix:
        if "SyntaxError" in diagnostics:
            lines = file_content.splitlines()
            for i, line in enumerate(lines):
                if "SyntaxError" in diagnostics and str(i+1) in diagnostics:
                    lines[i] = "# [AUTO-FIXED] " + line
            return "\n".join(lines)
        return file_content
    
    def test_solution(self, file_path):
        import subprocess
        try:
            result = subprocess.run(["python3", "-m", "py_compile", file_path], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def validate_repair(self, file_path):
        return self.test_solution(file_path)

class CodeEmbeddingIndex:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.file_snippets = []
    
    def build_index(self, file_map):
        if not VECTOR_SEARCH_AVAILABLE:
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
            # Make array contiguous before adding to index
            emb_matrix = np.ascontiguousarray(emb_matrix)
            self.index = faiss.IndexFlatL2(emb_matrix.shape[1])
            # Add the embedding matrix to the index
            faiss_add(self.index, emb_matrix)
    
    def search(self, query, top_k=5):
        if not VECTOR_SEARCH_AVAILABLE or not self.index or not np:
            return []
        q_emb = self.model.encode(query, convert_to_numpy=True).reshape(1, -1).astype('float32')
        # Make array contiguous and pass correct parameters to faiss
        q_emb = np.ascontiguousarray(q_emb)
        # Use the helper function to search
        distances, indices = faiss_search(self.index, q_emb, top_k)
        return [self.file_snippets[idx] for idx in indices[0] if idx < len(self.file_snippets)]

class SemanticCodeSearch:
    def __init__(self, file_map, model_name='all-MiniLM-L6-v2'):
        if not VECTOR_SEARCH_AVAILABLE:
            self.file_map = file_map
            self.model = None
            self.snippets = []
            self.embeddings = []
            self.emb_matrix = None
            return
            
        self.file_map = file_map
        self.model = SentenceTransformer(model_name)
        self.snippets = []
        self.embeddings = []
        for fpath, content in file_map.items():
            for i, line in enumerate(content.splitlines()):
                self.snippets.append({'file': fpath, 'line': i+1, 'text': line})
                self.embeddings.append(self.model.encode(line, convert_to_numpy=True))
        if self.embeddings and np:
            self.emb_matrix = np.vstack(self.embeddings).astype('float32')
            # Make array contiguous before adding to index
            self.emb_matrix = np.ascontiguousarray(self.emb_matrix)
        else:
            self.emb_matrix = None
    
    def search(self, query, top_k=5):
        if not VECTOR_SEARCH_AVAILABLE or self.emb_matrix is None or not np or not faiss or not self.model:
            return []
        q_emb = self.model.encode(query, convert_to_numpy=True).reshape(1, -1).astype('float32')
        q_emb = np.ascontiguousarray(q_emb)
        index = faiss.IndexFlatL2(self.emb_matrix.shape[1])
        # Use helper function to add vectors
        faiss_add(index, self.emb_matrix)
        # Use helper function to search
        D, I = faiss_search(index, q_emb, top_k)
        return [self.snippets[idx] for idx in I[0] if idx < len(self.snippets)]

# Orchestration pattern for automated debugging workflow
def automated_debugging_workflow(target_file):
    analyzer = CodeAnalyzerAgent()
    debugger = CodeDebuggerAgent()
    repairer = CodeRepairAgent()
    # 1. Scan and analyze
    file_map = analyzer.scan_repository()
    dep_graph = analyzer.build_dependency_graph(file_map)
    # 2. Focus on target file
    content = file_map.get(target_file)
    if not content:
        return f"File {target_file} not found."
    diagnostics = debugger.locate_bugs(content)
    root_cause = debugger.identify_root_cause(diagnostics)
    # 3. Generate repair
    fixed_content = repairer.generate_fix(content, diagnostics)
    tmp_path = "/tmp/_repaired.py"
    with open(tmp_path, "w") as f:
        f.write(fixed_content)
    # 4. Test and validate
    test_passed = repairer.test_solution(tmp_path)
    vector_memory.add(f"repair_{target_file}", json.dumps({"diagnostics": diagnostics, "root_cause": root_cause, "test_passed": test_passed}))
    return {"diagnostics": diagnostics, "root_cause": root_cause, "test_passed": test_passed}

def create_summarizer(specialization='summarization'): 
    """Create summarizer agent with summarization specialization"""
    return Agent('Summarizer', CEO_MODEL, specialization)

def create_test_generator(specialization='testing'): 
    """Create test generator agent with testing specialization"""
    return TestGeneratorAgent('TestGenerator', CEO_MODEL, specialization)

def create_dependency_agent(specialization='dependency_analysis'): 
    """Create dependency agent with dependency analysis specialization"""
    return DependencyAgent('DependencyAgent', CEO_MODEL, specialization)

class PerformanceProfilerAgent:
    def __init__(self):
        pass
    
    def profile_code(self, file_path, func_name=None):
        pr = cProfile.Profile()
        stats_output = io.StringIO()
        # Dynamically import and run the function
        import importlib.util
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
                return None  # Add fallback return value
                
            # Profile execution
            pr.enable()
            result = run()
            pr.disable()
            ps = pstats.Stats(pr, stream=stats_output).sort_stats('cumulative')
            ps.print_stats(20)
            cpu_profile = stats_output.getvalue()
            
            # Memory profiling using tracemalloc instead of memory_profiler
            _, mem_info = measure_memory_usage(run)
            
            return {"cpu_profile": cpu_profile, "mem_profile": mem_info, "result": result}
        except Exception as e:
            return {"cpu_profile": f"Error: {str(e)}", "mem_profile": [], "result": None}
    
    def find_hotspots(self, cpu_profile):
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
    
    def analyze_complexity(self, file_path):
        try:  # Add error handling for file reading
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                tokens = list(tokenize.generate_tokens(f.readline))
            func_complexities = {}
            func_name = None
            loop_stack = []
            for tok in tokens:
                if tok.type == tokenize.NAME and tok.string == 'def':
                    func_name = None
                elif tok.type == tokenize.NAME and func_name is None:
                    func_name = tok.string
                    func_complexities[func_name] = 1
                elif tok.type == tokenize.NAME and tok.string in ('for', 'while') and func_name:
                    func_complexities[func_name] *= 10  # crude: each loop increases complexity
            # Heuristic: >100 means likely O(n^2) or worse
            return {k: ('O(n^2) or worse' if v > 100 else 'O(n) or better') for k, v in func_complexities.items()}
        except Exception:
            return {}  # Return empty dict on error

class OptimizationSuggesterAgent:
    def __init__(self):
        pass
    def suggest(self, cpu_profile, mem_profile, complexity_report):
        suggestions = []
        for func, complexity in complexity_report.items():
            if 'O(n^2)' in complexity:
                suggestions.append(f"Function {func} has high complexity: {complexity}. Consider optimizing nested loops.")
        for hotspot in cpu_profile:
            suggestions.append(f"Hotspot: {hotspot['line']} (time: {hotspot['time']:.4f}s). Consider refactoring or memoization.")
        if max(mem_profile) - min(mem_profile) > 100:
            suggestions.append("High memory usage detected. Check for leaks or large data structures.")
        return suggestions

# Benchmarking framework
class BenchmarkingTool:
    def __init__(self):
        pass
    def benchmark(self, file_path, func_name=None, repeat=3):
        import timeit
        import importlib.util
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
            times = timeit.repeat(stmt=stmt, setup=setup, repeat=repeat, number=1, globals={'mod': mod})
            return {'min': min(times), 'max': max(times), 'avg': sum(times)/len(times)}
        except Exception as e:
            return {'error': str(e)}

class IntegratedCodebaseOptimizer:
    def __init__(self, root_dir="."):
        self.analyzer = CodeAnalyzerAgent(root_dir)
        self.debugger = CodeDebuggerAgent()
        self.repairer = CodeRepairAgent()
        self.profiler = PerformanceProfilerAgent()
        self.suggester = OptimizationSuggesterAgent()
        self.benchmarker = BenchmarkingTool()
    def optimize(self, target_file, func_name=None):
        # 1. Analyze codebase and build dependency graph
        file_map = self.analyzer.scan_repository()
        dep_graph = self.analyzer.build_dependency_graph(file_map)
        # 2. Prioritize high-impact components (most dependencies)
        # Count dependencies to calculate impact scores
        impact_scores = {}
        for f, deps in dep_graph.items():
            impact_scores[f] = len(deps) if deps else 0
        # Sort files by impact score (highest first)
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
                report["fixed_bugs"].append({"file": f, "diagnostics": diagnostics, "root_cause": root_cause, "test_passed": test_passed})
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

def pipeline_coordinator(target_file, func_name=None, root_dir="."):
    optimizer = IntegratedCodebaseOptimizer(root_dir)
    report = optimizer.optimize(target_file, func_name)
    # Unified reporting
    summary = []
    for bug in report["fixed_bugs"]:
        summary.append(f"Fixed bug in {bug['file']}: {bug['diagnostics']} (root cause: {bug['root_cause']}, test passed: {bug['test_passed']})")
    for f, perf in report["performance"].items():
        summary.append(f"Performance for {f}:\n  Original: {perf['original']['bench']}\n  Optimized: {perf['optimized']['bench']}\n  Suggestions: {perf['suggestions']}")
    return "\n".join(summary)
