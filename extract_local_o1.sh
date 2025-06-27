#!/bin/bash
# extract_local_o1.sh - Extract Local O1 components into current repository

set -e

SOURCE_REPO="https://github.com/UsernameTron/01-Local-Advanced-Model.git"
TEMP_DIR="temp_local_o1"
TARGET_DIR="local_o1_agents"

echo "🚀 Extracting Local O1 components..."

# Clean up any existing temp directory
rm -rf $TEMP_DIR

# Clone the source repository
echo "📥 Cloning source repository..."
git clone $SOURCE_REPO $TEMP_DIR

# Create target directory structure
echo "📁 Creating directory structure..."
mkdir -p $TARGET_DIR/{agents,orchestration,memory,models,config,utils,dashboard}

# Extract core agent files
echo "🤖 Extracting agent components..."
cp $TEMP_DIR/agents.py $TARGET_DIR/agents/
cp $TEMP_DIR/prompts.py $TARGET_DIR/agents/
cp $TEMP_DIR/config.py $TARGET_DIR/config/

# Extract orchestration system
echo "🎯 Extracting orchestration system..."
cp $TEMP_DIR/advanced_orchestrator.py $TARGET_DIR/orchestration/
cp $TEMP_DIR/orchestration_config.json $TARGET_DIR/orchestration/
cp $TEMP_DIR/task_complexity_analyzer.py $TARGET_DIR/orchestration/
cp $TEMP_DIR/orchestrator.py $TARGET_DIR/orchestration/

# Extract memory system
echo "🧠 Extracting memory system..."
cp $TEMP_DIR/vector_memory.py $TARGET_DIR/memory/

# Extract model management
echo "🔧 Extracting model management..."
cp $TEMP_DIR/distill_executor.py $TARGET_DIR/models/
cp $TEMP_DIR/distill_config.json $TARGET_DIR/models/
cp $TEMP_DIR/benchmark_executor.py $TARGET_DIR/models/
cp $TEMP_DIR/validate_executor_quality.py $TARGET_DIR/models/
cp $TEMP_DIR/parameter_search.py $TARGET_DIR/models/

# Extract utilities
echo "🛠️ Extracting utilities..."
cp $TEMP_DIR/utils.py $TARGET_DIR/utils/
cp $TEMP_DIR/system_test.py $TARGET_DIR/utils/

# Extract dashboard (optional)
echo "📊 Extracting dashboard..."
cp $TEMP_DIR/ui_dashboard.py $TARGET_DIR/dashboard/
cp $TEMP_DIR/ui_dashboard_config.json $TARGET_DIR/dashboard/

# Extract configuration files
echo "⚙️ Extracting configuration..."
cp $TEMP_DIR/requirements.txt $TARGET_DIR/
cp $TEMP_DIR/pyproject.toml $TARGET_DIR/

# Extract dataset for distillation (optional)
echo "📚 Extracting dataset..."
mkdir -p $TARGET_DIR/dataset
cp -r $TEMP_DIR/dataset/* $TARGET_DIR/dataset/ 2>/dev/null || echo "No dataset found"

# Extract scripts
echo "📜 Extracting scripts..."
mkdir -p $TARGET_DIR/scripts
cp -r $TEMP_DIR/scripts/* $TARGET_DIR/scripts/ 2>/dev/null || echo "No scripts found"

# Extract documentation
echo "📖 Extracting documentation..."
cp $TEMP_DIR/README_*.md $TARGET_DIR/ 2>/dev/null || echo "No README files found"
cp $TEMP_DIR/DEPLOYMENT_GUIDE.md $TARGET_DIR/ 2>/dev/null || echo "No deployment guide found"

# Create __init__.py files for Python modules
echo "🐍 Creating Python module files..."
touch $TARGET_DIR/agents/__init__.py
touch $TARGET_DIR/orchestration/__init__.py
touch $TARGET_DIR/memory/__init__.py
touch $TARGET_DIR/models/__init__.py
touch $TARGET_DIR/config/__init__.py
touch $TARGET_DIR/utils/__init__.py
touch $TARGET_DIR/dashboard/__init__.py

# Create a basic integration example
cat > $TARGET_DIR/example_integration.py << 'EOF'
"""
Basic integration example for Local O1 components
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import core components
from agents.agents import Agent, create_executor, create_ceo
from memory.vector_memory import vector_memory
from orchestration.advanced_orchestrator import run_advanced_pipeline

def main():
    """Example usage of Local O1 components"""
    print("🚀 Local O1 Integration Example")
    
    # Simple agent usage
    ceo = create_ceo()
    plan = ceo.run("Break down the task of building a web application")
    print(f"CEO Plan: {plan}")
    
    # Vector memory usage
    vector_memory.add("test task", "test output")
    cached = vector_memory.retrieve("test task")
    print(f"Cached result: {cached}")
    
    # Advanced orchestration
    result = run_advanced_pipeline("Design a simple API")
    print(f"Orchestration result: {result}")

if __name__ == "__main__":
    main()
EOF

# Create a requirements file with just the essentials
cat > $TARGET_DIR/requirements_minimal.txt << 'EOF'
# Essential dependencies for Local O1 integration
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
ollama>=0.1.0
streamlit>=1.28.0
psutil>=5.9.0
requests>=2.28.0
numpy>=1.24.0
pillow>=9.0.0
datasets>=2.14.0
matplotlib>=3.7.0
networkx>=3.1.0
pyvis>=0.3.2
memory-profiler>=0.61.0
torchaudio>=2.0.0
EOF

# Clean up
echo "🧹 Cleaning up..."
rm -rf $TEMP_DIR

echo "✅ Extraction complete!"
echo "📍 Files extracted to: $TARGET_DIR/"
echo ""
echo "🔧 Next steps:"
echo "1. cd $TARGET_DIR"
echo "2. pip install -r requirements_minimal.txt"
echo "3. python example_integration.py"
echo ""
echo "📚 Documentation available in:"
echo "   - README_*.md files"
echo "   - DEPLOYMENT_GUIDE.md"