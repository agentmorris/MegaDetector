import ast
import argparse
import re

def analyze_docstrings_for_modifications(filepath, tree, original_source_lines):
    """
    Analyzes docstrings and identifies lines to be modified by operating on original source lines.
    Returns a list of (line_number_to_change, new_line_content) tuples
    and a list of other findings. Line numbers are 1-indexed.
    """
    findings = []
    changes_to_apply = [] # List of (abs_line_num_1_indexed, new_line_string_without_newline)

    for item in tree.body:
        if isinstance(item, ast.ClassDef):
            for sub_item in item.body:
                if isinstance(sub_item, ast.FunctionDef):
                    process_function_node(sub_item, True, item, findings, changes_to_apply, original_source_lines)
        elif isinstance(item, ast.FunctionDef):
            process_function_node(item, False, None, findings, changes_to_apply, original_source_lines)

    return findings, changes_to_apply

def process_function_node(node, is_method_in_class, class_node, findings, changes_to_apply, original_source_lines):
    func_name = node.name

    if func_name.startswith('_'):
        return

    docstring_ast_constant_node = None
    if node.body and isinstance(node.body[0], ast.Expr) and \
       isinstance(node.body[0].value, ast.Constant) and \
       isinstance(node.body[0].value.value, str):
        docstring_ast_constant_node = node.body[0].value

    # --- Signature parsing (remains the same) ---
    sig_params = {}
    num_defaults = len(node.args.defaults)
    num_args = len(node.args.args)
    for i, arg in enumerate(node.args.args):
        sig_params[arg.arg] = {'has_default': (i >= num_args - num_defaults)}
    if node.args.vararg: sig_params[node.args.vararg.arg] = {'has_default': False}
    for i, arg_obj in enumerate(node.args.kwonlyargs):
        sig_params[arg_obj.arg] = {'has_default': (node.args.kw_defaults[i] is not None)}
    if node.args.kwarg: sig_params[node.args.kwarg.arg] = {'has_default': False}

    first_arg_name = node.args.args[0].arg if node.args.args else None
    is_classmethod = False
    if is_method_in_class:
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'classmethod':
                is_classmethod = True; break
    # --- End of signature parsing ---

    if not docstring_ast_constant_node:
        for param_name in sig_params:
            is_self = is_method_in_class and param_name == first_arg_name and param_name == 'self' and not is_classmethod
            is_cls = is_method_in_class and param_name == first_arg_name and param_name == 'cls' and is_classmethod
            if not (is_self or is_cls):
                 findings.append(f"Function '{func_name}', parameter '{param_name}': missing from docstring (no docstring found).")
        return

    start_line_idx = docstring_ast_constant_node.lineno - 1
    end_line_idx = docstring_ast_constant_node.end_lineno - 1

    if start_line_idx < 0 or end_line_idx >= len(original_source_lines):
        findings.append(f"Warning: Docstring line numbers for {func_name} out of bounds.")
        return

    docstring_literal_lines = [line.rstrip('\r\n') for line in original_source_lines[start_line_idx : end_line_idx+1]]

    in_args_section = False
    args_keyword = "Args:"
    line_param_pattern = re.compile(
        r"^(?P<indent>\s*)(?P<name>[\w_]+)\s*"
        r"(?:\((?P<type_group>[^)]*)\))?\s*:\s*"
        r"(?P<desc>.*)$"
    )
    parsed_doc_params = {}

    # Determine indentation of the "Args:" line to help identify end of section
    args_line_indent = None
    for i, current_line_content in enumerate(docstring_literal_lines):
        if args_keyword in current_line_content:
            match_indent = re.match(r"^(\s*)", current_line_content)
            if match_indent:
                args_line_indent = match_indent.group(1)
            break # Found Args line

    for i, current_line_content in enumerate(docstring_literal_lines):
        actual_file_line_num = start_line_idx + i + 1

        if args_keyword in current_line_content and not in_args_section: # Ensure it's the first time
            in_args_section = True
            continue
        if not in_args_section:
            continue

        current_line_stripped = current_line_content.strip()
        if not current_line_stripped: # Skip empty lines within Args
            continue

        # Check if the current line's indentation signals an end to the Args section
        if args_line_indent is not None:
            match_current_indent = re.match(r"^(\s*)", current_line_content)
            current_indent = match_current_indent.group(1) if match_current_indent else ""
            # If current line is less or equally indented than "Args:" line, and not empty, it's not a param (unless it's a continuation)
            # This simple check might not perfectly handle continuations of descriptions.
            if len(current_indent) < len(args_line_indent) + 1 : # Param lines must be more indented than "Args:" or at least one more level than docstring body
                 if not current_line_content.startswith(args_line_indent + " "): # allow same indent if it's not the start of args
                    if not (len(current_indent) > len(args_line_indent) and current_line_stripped): # allow more indented continuation lines
                        in_args_section = False # Stop processing args if line is not indented as a param/continuation
                        continue

        match = line_param_pattern.match(current_line_content)
        if not match:
            continue

        p_name = match.group('name')
        type_group_content = match.group('type_group')
        p_desc = match.group('desc')
        indent_str = match.group('indent')

        parsed_doc_params[p_name] = True
        original_type_spec_in_doc = type_group_content if type_group_content is not None else ""

        base_type_spec = original_type_spec_in_doc
        was_optional_present_in_doc = False

        if type_group_content is not None:
            temp_type_spec = type_group_content
            # More robust optional check and removal
            optional_pattern = r"\boptional\b"
            was_optional_present_in_doc = bool(re.search(optional_pattern, temp_type_spec, re.IGNORECASE))

            if was_optional_present_in_doc:
                # Remove "optional" and surrounding commas/spaces carefully
                parts = [p.strip() for p in temp_type_spec.split(',')]
                cleaned_parts = [p for p in parts if not re.fullmatch(optional_pattern, p, re.IGNORECASE)]
                base_type_spec = ", ".join(cleaned_parts)
            else:
                base_type_spec = temp_type_spec.strip() # Ensure no leading/trailing spaces

        base_type_spec = base_type_spec.strip() # Final strip, e.g. if it was just "optional"

        if not base_type_spec and was_optional_present_in_doc:
            findings.append(f"Function '{func_name}', parameter '{p_name}': type became empty after normalization (was '({type_group_content})'). Classified as missing type.")
            new_line_content = f"{indent_str}{p_name}: {p_desc}"
            if new_line_content != current_line_content:
                 changes_to_apply.append((actual_file_line_num, new_line_content))
            continue
        elif type_group_content is None or not type_group_content.strip() :
            findings.append(f"Function '{func_name}', parameter '{p_name}': missing type in docstring.")
            continue

        final_type_str = base_type_spec
        made_change_to_line = False
        optional_marker_to_add = "optional" # What we add

        if p_name in sig_params:
            param_has_default = sig_params[p_name]['has_default']
            should_be_optional = param_has_default

            if should_be_optional:
                if not was_optional_present_in_doc:
                    final_type_str = f"{base_type_spec}, {optional_marker_to_add}" if base_type_spec else optional_marker_to_add
                    findings.append(f"Function '{func_name}', parameter '{p_name}': added ', optional' (normalized from '({original_type_spec_in_doc})').")
                    made_change_to_line = True
                else: # Optional was present, ensure it's in the standard format
                    expected_type_str = f"{base_type_spec}, {optional_marker_to_add}" if base_type_spec else optional_marker_to_add
                    if original_type_spec_in_doc != expected_type_str : # Check if normalization changed anything
                        final_type_str = expected_type_str
                        findings.append(f"Function '{func_name}', parameter '{p_name}': normalized 'optional' placement/format (from '({original_type_spec_in_doc})').")
                        made_change_to_line = True
            else: # Should NOT be optional
                if was_optional_present_in_doc:
                    final_type_str = base_type_spec # Already stripped
                    findings.append(f"Function '{func_name}', parameter '{p_name}': removed ', optional' (normalized from '({original_type_spec_in_doc})').")
                    made_change_to_line = True
                elif base_type_spec != original_type_spec_in_doc.strip() : # No optional, but maybe spacing changes
                    final_type_str = base_type_spec
                    # findings.append(f"Function '{func_name}', parameter '{p_name}': type string normalized (from '({original_type_spec_in_doc})').")
                    made_change_to_line = True

            if made_change_to_line:
                if not final_type_str:
                    new_line_content = f"{indent_str}{p_name}: {p_desc}"
                else:
                    new_line_content = f"{indent_str}{p_name} ({final_type_str}): {p_desc}"

                if new_line_content != current_line_content:
                    changes_to_apply.append((actual_file_line_num, new_line_content))
        else:
            findings.append(f"Function '{func_name}', parameter '{p_name}': in docstring but not in signature.")

    for sig_p_name in sig_params:
        is_self = is_method_in_class and sig_p_name == first_arg_name and sig_p_name == 'self' and not is_classmethod
        is_cls = is_method_in_class and sig_p_name == first_arg_name and sig_p_name == 'cls' and is_classmethod
        if is_self or is_cls: continue
        if sig_p_name not in parsed_doc_params:
            args_section_exists_in_literal = any(args_keyword in line for line in docstring_literal_lines)
            if not args_section_exists_in_literal:
                 findings.append(f"Function '{func_name}', parameter '{sig_p_name}': missing from docstring (Args section not found or empty).")
            else:
                 findings.append(f"Function '{func_name}', parameter '{sig_p_name}': missing from docstring Args section.")

def main():
    parser = argparse.ArgumentParser(description="Refines 'optional' in Python function docstrings using direct string manipulation.")
    parser.add_argument("filepath", help="Path to the Python file to analyze and modify.")
    args = parser.parse_args()

    try:
        with open(args.filepath, 'r', encoding='utf-8') as file:
            source_code = file.read()
        original_lines_with_endings = source_code.splitlines(True)
        tree = ast.parse(source_code, filename=args.filepath)
    except FileNotFoundError:
        print(f"Error: File not found at '{args.filepath}'")
        return
    except SyntaxError as e:
        print(f"Error: Could not parse Python file '{args.filepath}': {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading/parsing '{args.filepath}': {e}")
        return

    findings, changes_to_apply = analyze_docstrings_for_modifications(args.filepath, tree, original_lines_with_endings)

    change_log_messages = {f for f in findings if "added ', optional'" in f or "removed ', optional'" in f or "normalized 'optional' placement" in f or "type string normalized" in f}
    other_findings = [f for f in findings if f not in change_log_messages]

    if other_findings:
        for finding in sorted(list(set(other_findings))):
            print(finding)

    if changes_to_apply:
        for msg in sorted(list(change_log_messages)):
            print(msg)

        modified_lines = list(original_lines_with_endings)

        for line_num_1_indexed, new_content_for_line in changes_to_apply:
            original_line_idx = line_num_1_indexed - 1
            if 0 <= original_line_idx < len(modified_lines):
                original_ending = ''
                stripped_original = original_lines_with_endings[original_line_idx].rstrip('\r\n')
                original_ending = original_lines_with_endings[original_line_idx][len(stripped_original):]

                if not original_ending and original_line_idx < len(original_lines_with_endings) - 1:
                    original_ending = '\n'
                elif not original_ending and original_line_idx == len(original_lines_with_endings) -1:
                     pass

                modified_lines[original_line_idx] = new_content_for_line + original_ending
            else:
                print(f"Error: Calculated line number {line_num_1_indexed} for change is out of bounds for file {args.filepath}")

        try:
            with open(args.filepath, 'w', encoding='utf-8') as file:
                file.writelines(modified_lines)
            print(f"File '{args.filepath}' was modified.")
        except Exception as e:
            print(f"Error writing modified code to file: {e}")

    elif not other_findings and not changes_to_apply:
        print("No docstring issues or modifications needed/made.")

if __name__ == "__main__":
    main()
