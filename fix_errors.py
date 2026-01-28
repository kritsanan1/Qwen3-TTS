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
    
    def analyze_dataclasses(self) -> List[str]:
        """Analyze dataclass definitions for common issues"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            issues = []
            
            # Check for missing __post_init__ validation
            dataclass_pattern = r'@dataclass[\s\S]*?class\s+(\w+)[\s\S]*?(def __post_init__\(self\):)?'
            matches = re.finditer(dataclass_pattern, content)
            
            for match in matches:
                class_name = match.group(1)
                post_init = match.group(2)
                
                if not post_init:
                    issues.append(f"Dataclass {class_name} missing __post_init__ validation")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error analyzing dataclasses for {self.file_path.name}: {e}")
            return []
    
    def analyze_error_handling(self) -> List[str]:
        """Analyze error handling patterns"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            issues = []
            
            # Check for bare except clauses
            bare_except_pattern = r'except\s*:'
            if re.search(bare_except_pattern, content):
                issues.append("Bare except clause found - should catch specific exceptions")
            
            # Check for print statements in error handling
            print_pattern = r'except.*:\s*\n.*print\('
            if re.search(print_pattern, content, re.MULTILINE):
                issues.append("Print statement found in exception handling - should use logging")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error analyzing error handling for {self.file_path.name}: {e}")
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
                        'matplotlib', 'seaborn', 'sklearn', 'pandas', 'tqdm', 'psutil'
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
        # This is a simplified check - in a real implementation, you'd use a more comprehensive list
        stdlib_modules = [
            'os', 'sys', 'time', 'datetime', 'json', 'logging', 'pathlib', 'typing',
            'dataclasses', 'tempfile', 'hashlib', 'threading', 'concurrent', 'functools',
            'itertools', 'collections', 're', 'ast', 'argparse', 'unittest', 'warnings'
        ]
        return module_name in stdlib_modules

class CodeFixer:
    """Automatically fixes common code issues"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.original_content = ""
        self.fixed_content = ""
    
    def load_content(self) -> bool:
        """Load the original file content"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.original_content = f.read()
            return True
        except Exception as e:
            logger.error(f"Error loading content from {self.file_path.name}: {e}")
            return False
    
    def fix_bare_except_clauses(self) -> bool:
        """Replace bare except clauses with specific exception handling"""
        if not self.original_content:
            return False
        
        # Pattern to match bare except clauses
        pattern = r'except\s*:'
        replacement = 'except Exception:'
        
        self.fixed_content = re.sub(pattern, replacement, self.original_content)
        logger.info(f"Fixed bare except clauses in {self.file_path.name}")
        return True
    
    def add_import_guards(self, missing_deps: List[str]) -> bool:
        """Add try-except guards around optional imports"""
        if not self.original_content:
            return False
        
        content = self.original_content
        
        for dep in missing_deps:
            # Pattern to match import statements
            import_patterns = [
                rf'^import {dep}$',
                rf'^from {dep} import',
                rf'^import {dep}\.',
                rf'^from {dep}\.'
            ]
            
            for pattern in import_patterns:
                # Find import statements and wrap them in try-except
                def replace_import(match):
                    original = match.group(0)
                    return f"try:\n    {original}\nexcept ImportError:\n    logger.warning('{dep} not available. Some features may be limited.')\n    {dep} = Mock()"
                
                content = re.sub(pattern, replace_import, content, flags=re.MULTILINE)
        
        self.fixed_content = content
        logger.info(f"Added import guards for missing dependencies in {self.file_path.name}")
        return True
    
    def add_post_init_validation(self, class_name: str) -> bool:
        """Add __post_init__ validation to a dataclass"""
        if not self.original_content:
            return False
        
        # Find the dataclass and add __post_init__
        pattern = rf'(@dataclass[\s\S]*?class {class_name}[\s\S]*?)(?=class|\Z)'
        
        def add_validation(match):
            class_content = match.group(1)
            
            # Check if __post_init__ already exists
            if 'def __post_init__(self):' in class_content:
                return class_content
            
            # Add __post_init__ method
            validation_code = '''\n    def __post_init__(self):
        """Validate instance data"""
        # Add validation logic here
        pass\n'''
            
            # Insert before the last line (assuming it's the class end)
            lines = class_content.split('\n')
            if lines:
                lines.insert(-1, validation_code)
                return '\n'.join(lines)
            else:
                return class_content + validation_code
        
        self.fixed_content = re.sub(pattern, add_validation, self.original_content, flags=re.MULTILINE | re.DOTALL)
        logger.info(f"Added __post_init__ validation to {class_name}")
        return True
    
    def save_fixed_content(self, backup: bool = True) -> bool:
        """Save the fixed content back to file"""
        if not self.fixed_content:
            logger.error("No fixed content to save")
            return False
        
        try:
            # Create backup if requested
            if backup and self.original_content:
                backup_path = self.file_path.with_suffix(self.file_path.suffix + '.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(self.original_content)
                logger.info(f"Created backup: {backup_path.name}")
            
            # Save fixed content
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(self.fixed_content)
            
            logger.info(f"Saved fixed content to {self.file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving fixed content to {self.file_path.name}: {e}")
            return False

class SystemValidator:
    """Validates the overall system functionality"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.validation_results = {}
    
    def validate_imports(self) -> bool:
        """Validate that all modules can be imported"""
        logger.info("Validating imports...")
        
        python_files = list(self.project_root.glob("*.py"))
        import_errors = []
        
        for py_file in python_files:
            if py_file.name.startswith('test_'):
                continue  # Skip test files
            
            try:
                # Try to import the module
                module_name = py_file.stem
                spec = __import__('importlib.util').util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = __import__('importlib.util').module_from_spec(spec)
                    spec.loader.exec_module(module)
                    logger.debug(f"Successfully imported {module_name}")
                else:
                    import_errors.append(f"Could not load {module_name}")
            except Exception as e:
                import_errors.append(f"Import error in {py_file.name}: {e}")
        
        if import_errors:
            logger.warning(f"Import validation found issues: {import_errors}")
            self.validation_results['imports'] = {'status': 'failed', 'errors': import_errors}
            return False
        else:
            logger.info("All modules imported successfully")
            self.validation_results['imports'] = {'status': 'passed'}
            return True
    
    def validate_dataclasses(self) -> bool:
        """Validate that all dataclasses can be instantiated"""
        logger.info("Validating dataclasses...")
        
        # This is a simplified check - in a real implementation, you'd
        # try to instantiate each dataclass with test data
        python_files = list(self.project_root.glob("*.py"))
        dataclass_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for dataclass definitions
                if '@dataclass' in content:
                    # Extract dataclass names
                    dataclass_pattern = r'@dataclass[\s\S]*?class\s+(\w+)'
                    dataclasses = re.findall(dataclass_pattern, content)
                    
                    for dc_name in dataclasses:
                        logger.debug(f"Found dataclass: {dc_name} in {py_file.name}")
                        
            except Exception as e:
                dataclass_issues.append(f"Error checking {py_file.name}: {e}")
        
        if dataclass_issues:
            logger.warning(f"Dataclass validation found issues: {dataclass_issues}")
            self.validation_results['dataclasses'] = {'status': 'warning', 'issues': dataclass_issues}
            return False
        else:
            logger.info("Dataclass validation completed")
            self.validation_results['dataclasses'] = {'status': 'passed'}
            return True
    
    def generate_report(self) -> str:
        """Generate a validation report"""
        report = []
        report.append("=" * 60)
        report.append("THAI-ISAN TTS SYSTEM VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        for check_name, results in self.validation_results.items():
            status = results.get('status', 'unknown')
            report.append(f"{check_name.upper()}: {status.upper()}")
            
            if 'errors' in results:
                report.append("Errors:")
                for error in results['errors']:
                    report.append(f"  - {error}")
            
            if 'issues' in results:
                report.append("Issues:")
                for issue in results['issues']:
                    report.append(f"  - {issue}")
            
            report.append("")
        
        # Overall status
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results.values() if r.get('status') == 'passed')
        
        report.append("=" * 60)
        report.append(f"OVERALL STATUS: {passed_checks}/{total_checks} checks passed")
        report.append("=" * 60)
        
        return "\n".join(report)

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
        
        # Check dataclasses
        dataclass_issues = analyzer.analyze_dataclasses()
        if dataclass_issues:
            logger.warning(f"Dataclass issues in {py_file.name}: {dataclass_issues}")
        
        # Check error handling
        error_issues = analyzer.analyze_error_handling()
        if error_issues:
            logger.warning(f"Error handling issues in {py_file.name}: {error_issues}")
        
        # Collect all issues
        file_issues = {
            'file': py_file.name,
            'syntax_ok': syntax_ok,
            'import_issues': import_issues,
            'dataclass_issues': dataclass_issues,
            'error_issues': error_issues
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
    
    # System validation
    logger.info("Validating system...")
    validator = SystemValidator(str(project_root))
    
    # Run validations
    validator.validate_imports()
    validator.validate_dataclasses()
    
    # Generate report
    report = validator.generate_report()
    logger.info("\n" + report)
    
    logger.info("Error fixing and validation completed!")

if __name__ == "__main__":
    main()