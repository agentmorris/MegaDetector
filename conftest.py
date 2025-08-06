import os
import ast
import sys # noqa

def has_test_functions(filepath):
    """
    Parse a Python file and check whether it contains functions starting with 'test_'.
    Also checks for test methods in classes.
    """

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=filepath)

        for node in ast.walk(tree):

            # Check for standalone test functions
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                return True

            # Check for test methods in classes
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                        return True

        return False

    except Exception as e:
        # Log the error if you want to debug
        print(f"Warning: Could not parse {filepath}: {e}")
        return False


def pytest_ignore_collect(collection_path, config):
    """
    Tell pytest which files to ignore
    """

    s_path = str(collection_path)

    if 'job_status' in s_path:
        print('****found status: {}, {} ***'.format(s_path,format(collection_path.suffix)))

    # Ignore legacy folders
    if 'api_core' in s_path:
        return True
    if 'classification' in s_path:
        return True
    # if 'md_tests' in s_path:
    #    return True

    # Let pytest figure out what to do with non-py files, including folders
    if collection_path.suffix != '.py':
        return False

    # Ignore cache files
    if '__pycache__' in s_path:
        return True

    # Don't ignore Python files that have test functions
    if has_test_functions(collection_path):
        print('Found test functions in {}'.format(s_path))
        return False

    # Ignore other .py files
    return True


def pytest_collection_modifyitems(config, items):
    """
    Log items discovered
    """

    print(f"\nCustom collection found {len(items)} test items:")
    for item in items:
        print(f"  {item.nodeid}")


def test_environment_debug():
    print(f"PYTHONPATH from os.environ: {os.environ.get('PYTHONPATH', 'NOT SET')}")
    print(f"sys.path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
