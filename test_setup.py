"""
Test script to verify system setup and dependencies
Run this to check if everything is installed correctly
"""

import sys
import subprocess

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name} NOT installed")
        return False

def check_ollama():
    """Check if Ollama is running and models are available"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print("✓ Ollama is running")
            
            # Check for specific models
            models_to_check = ['mistral', 'qwen2.5', 'llama3']
            found_models = []
            
            for model in models_to_check:
                if model in result.stdout:
                    print(f"  ✓ {model} model available")
                    found_models.append(model)
                else:
                    print(f"  ✗ {model} model NOT installed (run: ollama pull {model})")
            
            return len(found_models) > 0
        else:
            print("✗ Ollama is not running")
            print("  Start Ollama: ollama serve")
            return False
    except FileNotFoundError:
        print("✗ Ollama is not installed")
        print("  Install from: https://ollama.com/download")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {str(e)}")
        return False

def check_files():
    """Check if required files exist"""
    import os
    
    required_files = [
        'pdf_rag_system.py',
        'evaluation_metrics.py',
        'visualization.py',
        'run_evaluation.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✓ {filename} exists")
        else:
            print(f"✗ {filename} missing")
            all_exist = False
    
    return all_exist

def check_directories():
    """Check if required directories exist"""
    import os
    
    dirs = ['pdfs']
    
    for dirname in dirs:
        if os.path.exists(dirname):
            print(f"✓ {dirname}/ directory exists")
            
            # Check for PDFs
            if dirname == 'pdfs':
                pdf_files = [f for f in os.listdir(dirname) if f.endswith('.pdf')]
                if pdf_files:
                    print(f"  ✓ Found {len(pdf_files)} PDF file(s)")
                else:
                    print(f"  ⚠️  No PDF files found (add PDFs to test)")
        else:
            print(f"✗ {dirname}/ directory missing")
            os.makedirs(dirname, exist_ok=True)
            print(f"  Created {dirname}/ directory")

def main():
    """Run all checks"""
    print("=" * 70)
    print("PDF RAG System - Setup Verification")
    print("=" * 70)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Python 3.8 or higher is recommended")
    else:
        print("✓ Python version is compatible")
    
    # Check files
    print("\n" + "-" * 70)
    print("Checking Required Files:")
    print("-" * 70)
    files_ok = check_files()
    
    # Check directories
    print("\n" + "-" * 70)
    print("Checking Directories:")
    print("-" * 70)
    check_directories()
    
    # Check dependencies
    print("\n" + "-" * 70)
    print("Checking Python Dependencies:")
    print("-" * 70)
    
    dependencies = [
        ('langchain', 'langchain'),
        ('sentence_transformers', 'sentence-transformers'),
        ('faiss', 'faiss-cpu'),
        ('nltk', 'nltk'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('evaluate', 'evaluate'),
    ]
    
    all_installed = True
    for module, package in dependencies:
        if not check_import(module, package):
            all_installed = False
    
    # Check Ollama
    print("\n" + "-" * 70)
    print("Checking Ollama:")
    print("-" * 70)
    ollama_ok = check_ollama()
    
    # Download NLTK data
    print("\n" + "-" * 70)
    print("Checking NLTK Data:")
    print("-" * 70)
    try:
        import nltk
        required_data = ['punkt', 'wordnet', 'omw-1.4']
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' else f'corpora/{data_name}')
                print(f"✓ NLTK {data_name} available")
            except LookupError:
                print(f"⚠️  NLTK {data_name} not found, downloading...")
                nltk.download(data_name, quiet=True)
                print(f"✓ Downloaded {data_name}")
    except Exception as e:
        print(f"✗ Error checking NLTK data: {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if files_ok and all_installed:
        print("✓ All required files and dependencies are installed")
    else:
        print("⚠️  Some dependencies are missing")
        print("   Run: pip install -r requirements.txt")
    
    if ollama_ok:
        print("✓ Ollama is configured and models are available")
    else:
        print("⚠️  Ollama needs setup")
        print("   1. Install Ollama: https://ollama.com/download")
        print("   2. Start Ollama: ollama serve")
        print("   3. Pull models: ollama pull mistral && ollama pull qwen2.5 && ollama pull llama3")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    if files_ok and all_installed and ollama_ok:
        print("✓ System is ready!")
        print("  1. Add PDF files to pdfs/ directory")
        print("  2. Run: python run_evaluation.py")
    else:
        print("Complete the setup:")
        if not (files_ok and all_installed):
            print("  1. Install dependencies: pip install -r requirements.txt")
        if not ollama_ok:
            print("  2. Setup Ollama and install models")
        print("  3. Re-run this test: python test_setup.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
