# Agent Development

A multi-agent AI orchestration system using Python for distributed task processing, with support for multimodal inputs like images and audio.

## Features

- Multiple specialized AI agents working together
- Image analysis and processing via `ImageAgent`
- Audio transcription and analysis via `AudioAgent`
- Code analysis, debugging, and repair
- Vector memory for efficient knowledge retrieval
- Robust error handling and dependency management

## Requirements

- Python 3.8+
- PyTorch
- FAISS for vector indexing
- Transformers (Hugging Face)
- torchaudio
- sentence-transformers

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main orchestrator: `python -m local_o1_agents.orchestrator`

## Architecture

The system uses a modular architecture with specialized agents:

- `Agent`: Base agent class
- `TestGeneratorAgent`: Generates test cases
- `DependencyAgent`: Analyzes code dependencies
- `ExecutorWithFallback`: Executes code with fallback mechanisms
- `ImageAgent`: Handles image analysis using CLIP
- `AudioAgent`: Processes audio using Whisper
- `CodeAnalyzerAgent`: Analyzes code repositories
- `CodeDebuggerAgent`: Debugs and traces code execution
- `CodeEmbeddingIndex`: Vector database for code
- `CodeRepairAgent`: Fixes code issues
- `PerformanceProfilerAgent`: Profiles code performance

## License

MIT
