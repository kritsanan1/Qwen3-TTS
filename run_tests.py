"""
Comprehensive test runner for Thai-Isan TTS System
Runs all unit tests, integration tests, and performance benchmarks
"""

import unittest
import sys
import os
import argparse
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_unit_tests():
    """Run all unit tests"""
    logger.info("Running unit tests...")
    
    # Discover and run unit tests
    loader = unittest.TestLoader()
    start_dir = str(project_root / 'tests' / 'unit')
    
    try:
        suite = loader.discover(start_dir, pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        logger.info(f"Unit tests completed: {result.testsRun} tests run")
        if result.wasSuccessful():
            logger.info("All unit tests passed!")
            return True
        else:
            logger.error(f"Unit tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
            return False
    except Exception as e:
        logger.error(f"Error running unit tests: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    logger.info("Running integration tests...")
    
    # Discover and run integration tests
    loader = unittest.TestLoader()
    start_dir = str(project_root / 'tests' / 'integration')
    
    try:
        suite = loader.discover(start_dir, pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        logger.info(f"Integration tests completed: {result.testsRun} tests run")
        if result.wasSuccessful():
            logger.info("All integration tests passed!")
            return True
        else:
            logger.error(f"Integration tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
            return False
    except Exception as e:
        logger.error(f"Error running integration tests: {e}")
        return False

def run_performance_tests():
    """Run all performance tests"""
    logger.info("Running performance tests...")
    
    # Discover and run performance tests
    loader = unittest.TestLoader()
    start_dir = str(project_root / 'tests' / 'performance')
    
    try:
        suite = loader.discover(start_dir, pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        logger.info(f"Performance tests completed: {result.testsRun} tests run")
        if result.wasSuccessful():
            logger.info("All performance tests passed!")
            return True
        else:
            logger.error(f"Performance tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
            return False
    except Exception as e:
        logger.error(f"Error running performance tests: {e}")
        return False

def run_all_tests():
    """Run all test suites"""
    logger.info("Starting comprehensive test suite...")
    start_time = time.time()
    
    results = {
        'unit': False,
        'integration': False,
        'performance': False
    }
    
    # Run unit tests
    logger.info("=" * 50)
    logger.info("UNIT TESTS")
    logger.info("=" * 50)
    results['unit'] = run_unit_tests()
    
    # Run integration tests
    logger.info("=" * 50)
    logger.info("INTEGRATION TESTS")
    logger.info("=" * 50)
    results['integration'] = run_integration_tests()
    
    # Run performance tests
    logger.info("=" * 50)
    logger.info("PERFORMANCE TESTS")
    logger.info("=" * 50)
    results['performance'] = run_performance_tests()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Unit Tests: {'PASSED' if results['unit'] else 'FAILED'}")
    logger.info(f"Integration Tests: {'PASSED' if results['integration'] else 'FAILED'}")
    logger.info(f"Performance Tests: {'PASSED' if results['performance'] else 'FAILED'}")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    all_passed = all(results.values())
    logger.info(f"Overall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

def run_specific_test(test_path):
    """Run a specific test file or test method"""
    logger.info(f"Running specific test: {test_path}")
    
    try:
        # Try to import and run the specific test
        if ':' in test_path:
            # Test method format: module.Class.method
            module_path, class_method = test_path.split(':', 1)
            module_name = module_path.replace('/', '.').replace('.py', '')
            
            # Import the module
            __import__(module_name)
            module = sys.modules[module_name]
            
            # Get the test class and method
            if '.' in class_method:
                class_name, method_name = class_method.split('.', 1)
                test_class = getattr(module, class_name)
                test_instance = test_class()
                test_method = getattr(test_instance, method_name)
                
                # Run the specific test method
                test_method()
                logger.info(f"Test {test_path} completed successfully")
                return True
        else:
            # Test file format
            test_path = str(project_root / test_path)
            loader = unittest.TestLoader()
            suite = loader.discover(test_path, pattern='test_*.py')
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            return result.wasSuccessful()
            
    except Exception as e:
        logger.error(f"Error running specific test {test_path}: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    logger.info("Checking dependencies...")
    
    required_modules = [
        'numpy', 'scipy', 'sklearn', 'pandas', 'tqdm'
    ]
    
    optional_modules = [
        'torch', 'librosa', 'soundfile', 'pythainlp', 'pesq', 'stoi', 
        'matplotlib', 'seaborn', 'psutil'
    ]
    
    available_modules = []
    missing_required = []
    missing_optional = []
    
    # Check required modules
    for module in required_modules:
        try:
            __import__(module)
            available_modules.append(module)
        except ImportError:
            missing_required.append(module)
    
    # Check optional modules
    for module in optional_modules:
        try:
            __import__(module)
            available_modules.append(module)
        except ImportError:
            missing_optional.append(module)
    
    # Report results
    logger.info(f"Available modules ({len(available_modules)}): {', '.join(available_modules)}")
    
    if missing_required:
        logger.error(f"Missing required modules ({len(missing_required)}): {', '.join(missing_required)}")
        return False
    else:
        logger.info("All required modules are available")
    
    if missing_optional:
        logger.warning(f"Missing optional modules ({len(missing_optional)}): {', '.join(missing_optional)}")
        logger.warning("Some features may be limited, but core functionality should work")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Thai-Isan TTS System Test Runner')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--specific', type=str, help='Run specific test (format: path/to/test.py:ClassName.method_name)')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("Missing required dependencies. Please install required packages.")
        return 1
    
    # Handle specific commands
    if args.check_deps:
        return 0
    
    if args.specific:
        success = run_specific_test(args.specific)
        return 0 if success else 1
    
    # Determine which tests to run
    if args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.performance:
        success = run_performance_tests()
    else:
        # Run all tests by default
        success = run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())