"""
Error fixing and validation script for Thai-Isan TTS System
Identifies and fixes common issues in the codebase
"""

import ast
import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """Analyzes Python code for common issues"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.issues = []
        self.fixed_code = None
    
    def analyze_syntax(self) -> bool:
        """Check for syntax errors"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            ast.parse(content)
            logger.info(f"Syntax check passed for {self.file_path.name}")
            return True
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {self.file_path.name}: {e}")
            self.issues.append(f"Syntax error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error analyzing syntax for {self.file_path.name}: {e}")
            return False
    
    def analyze_imports(self) -> List[str]:
        """Analyze import statements for potential issues"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find import statements
            import_pattern = r'^(import|from)\s+([\w\.]+)'
            imports = re.findall(import_pattern, content, re.MULTILINE)
            
            issues = []
            for import_type, module_name in imports:
                # Check for relative imports that might cause issues
                if module_name.startswith('.'):
                    issues.append(f"Relative import found: {import_type} {module_name}")
                
                # Check for potentially problematic imports
                problematic_modules = ['tkinter', 'PyQt', 'wx']
                for problematic in problematic_modules:
                    if problematic in module_name:
                        issues.append(f"Potentially problematic GUI import: {module_name}")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error analyzing imports for {self.file_path.name}: {e}")
            return []

class DependencyChecker:
    """Checks for missing dependencies and suggests fixes"""
    
    def __init__(self):
        self.missing_deps = []
        self.optional_deps = []
    
    def check_imports(self, file_path: str) -> Dict[str, List[str]]:
        """Check imports in a file for missing dependencies"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all import statements
            import_pattern = r'^(?:import|from)\s+([\w\.]+)'
            imports = re.findall(import_pattern, content, re.MULTILINE)
            
            missing_required = []
            missing_optional = []
            
            for module_name in imports:
                # Handle 'from module import something'
                base_module = module_name.split('.')[0]
                
                # Skip standard library modules
                if self._is_standard_library(base_module):
                    continue
                
                # Check if module can be imported
                try:
                    __import__(base_module)
                except ImportError:
                    # Categorize as required or optional
                    optional_modules = [
                        'torch', 'librosa', 'soundfile', 'pythainlp', 'pesq', 'stoi',
                        'matplotlib', 'seaborn', 'sklearn', 'pathlib', 'tqdm', 'psutil'
                    ]
                    
                    if base_module in optional_modules:
                        missing_optional.append(base_module)
                    else:
                        missing_required.append(base_module)
            
            return {
                'missing_required': missing_required,
                'missing_optional': missing_optional,
                'all_imports': list(set(imports))
            }
            
        except Exception as e:
            logger.error(f"Error checking imports for {file_path}: {e}")
            return {'missing_required': [], 'missing_optional': [], 'all_imports': []}
    
    def _is_standard_library(self, module_name: str) -> bool:
        """Check if a module is part of Python standard library"""
        stdlib_modules = [
            'os', 'sys', 'time', 'datetime', 'json', 'logging', 'pathlib', 'typing',
            'dataclasses', 'tempfile', 'hashlib', 'threading', 'concurrent', 'functools',
            'itertools', 'collections', 're', 'ast', 'argparse', 'unittest', 'warnings'
        ]
        return module_name in stdlib_modules

def main():
    """Main function"""
    project_root = Path(__file__).parent
    
    logger.info("Starting Thai-Isan TTS System error fixing and validation...")
    logger.info(f"Project root: {project_root}")
    
    # Find all Python files to analyze
    python_files = list(project_root.glob("*.py"))
    logger.info(f"Found {len(python_files)} Python files to analyze")
    
    # Analyze each file
    all_issues = []
    for py_file in python_files:
        if py_file.name.startswith('test_') or py_file.name == 'run_tests.py':
            continue  # Skip test files and test runner
        
        logger.info(f"Analyzing {py_file.name}...")
        
        # Analyze the file
        analyzer = CodeAnalyzer(str(py_file))
        
        # Check syntax
        syntax_ok = analyzer.analyze_syntax()
        if not syntax_ok:
            logger.error(f"Syntax errors found in {py_file.name}")
        
        # Check imports
        import_issues = analyzer.analyze_imports()
        if import_issues:
            logger.warning(f"Import issues in {py_file.name}: {import_issues}")
        
        # Collect all issues
        file_issues = {
            'file': py_file.name,
            'syntax_ok': syntax_ok,
            'import_issues': import_issues
        }
        all_issues.append(file_issues)
    
    # Check dependencies
    logger.info("Checking dependencies...")
    dependency_checker = DependencyChecker()
    
    # Check imports in main files
    for py_file in python_files:
        if py_file.name.startswith('test_'):
            continue
        
        dep_info = dependency_checker.check_imports(str(py_file))
        if dep_info['missing_required'] or dep_info['missing_optional']:
            logger.warning(f"Dependency issues in {py_file.name}:")
            if dep_info['missing_required']:
                logger.error(f"  Missing required: {dep_info['missing_required']}")
            if dep_info['missing_optional']:
                logger.warning(f"  Missing optional: {dep_info['missing_optional']}")
    
    logger.info("Error fixing and validation completed!")

if __name__ == "__main__":
    main()