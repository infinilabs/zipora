"""Shared Rust source code parser and token scanner for audit scripts.

Provides robust test-scope tracking, comment/string stripping, and attribute
parsing across non-test and test code.
"""

import os
import re

TEST_DIR_PATTERNS = ('/tests/', '\\tests\\', '/benches/', '\\benches\\')

def is_test_file(filepath):
    """Check if the file itself is a dedicated test file or benchmark."""
    norm = os.path.normpath(filepath).replace('\\', '/')
    if any(p in norm for p in ('/tests/', '/benches/')):
        return True
    if norm.endswith('_tests.rs') or norm.endswith('/tests.rs') or norm.endswith('/test.rs'):
        return True
    return False

def scan_rust_file(filepath):
    """Scan a Rust file and yield parsed lines with test-scope and comment metadata.

    Yields dicts with:
      - 'line_idx': 0-based line index
      - 'line_num': 1-based line number
      - 'raw_line': raw line string
      - 'clean_line': code text with comments and string literal contents stripped
      - 'in_test': bool, True if line is inside a test function/module or test file
      - 'doc_comments': list of preceding doc comments (/// or //!)
      - 'comments_above': list of preceding line/doc comments within the last 20 lines
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    if is_test_file(filepath):
        for idx, line in enumerate(lines):
            yield {
                'line_idx': idx,
                'line_num': idx + 1,
                'raw_line': line,
                'clean_line': '',
                'in_test': True,
                'doc_comments': [],
                'comments_above': [],
            }
        return

    in_block_comment = 0
    brace_stack = []  # list of bool: is_test
    pending_test = False
    recent_comments = []  # list of (line_idx, text)
    recent_doc_comments = []  # list of (line_idx, text)

    for idx, line in enumerate(lines):
        line_num = idx + 1
        raw_stripped = line.strip()

        # Check for pure comment lines
        is_pure_comment = False
        if raw_stripped.startswith('///') or raw_stripped.startswith('//!'):
            recent_doc_comments.append((idx, raw_stripped))
            recent_comments.append((idx, raw_stripped))
            is_pure_comment = True
        elif raw_stripped.startswith('//'):
            recent_comments.append((idx, raw_stripped))
            is_pure_comment = True
        elif not raw_stripped:
            # Empty line; prune comments older than 20 lines
            recent_comments = [(i, t) for (i, t) in recent_comments if idx - i <= 20]
            recent_doc_comments = [(i, t) for (i, t) in recent_doc_comments if idx - i <= 20]

        if is_pure_comment:
            in_test = (len(brace_stack) > 0 and brace_stack[-1]) or pending_test
            docs = [t for (c_idx, t) in recent_doc_comments if idx - c_idx <= 20]
            comms = [t for (c_idx, t) in recent_comments if idx - c_idx <= 20]
            yield {
                'line_idx': idx,
                'line_num': line_num,
                'raw_line': line,
                'clean_line': '',
                'in_test': in_test,
                'doc_comments': docs,
                'comments_above': comms,
            }
            continue

        # Parse character-by-character to strip string contents and handle block comments
        i = 0
        n = len(line)
        clean = []

        while i < n:
            if in_block_comment > 0:
                if line[i:i+2] == '*/':
                    in_block_comment -= 1
                    i += 2
                elif line[i:i+2] == '/*':
                    in_block_comment += 1
                    i += 2
                else:
                    i += 1
                continue

            if line[i:i+2] == '/*':
                in_block_comment += 1
                i += 2
                continue

            if line[i:i+3] in ('///', '//!'):
                doc_text = line[i:].strip()
                recent_doc_comments.append((idx, doc_text))
                recent_comments.append((idx, doc_text))
                break

            if line[i:i+2] == '//':
                recent_comments.append((idx, line[i:].strip()))
                break

            # Handle string literal
            if line[i] == '"':
                clean.append('"')
                i += 1
                while i < n:
                    if line[i] == '\\':
                        i += 2
                    elif line[i] == '"':
                        clean.append('"')
                        i += 1
                        break
                    else:
                        i += 1
                continue

            if line[i] == "'":
                # Char literal vs lifetime
                if i + 2 < n and line[i+2] == "'":
                    i += 3
                    continue
                elif i + 3 < n and line[i+1] == '\\' and line[i+3] == "'":
                    i += 4
                    continue

            clean.append(line[i])
            i += 1

        clean_line = ''.join(clean).strip()

        # Check for inner attribute #![cfg(test)] marking the whole file as test
        if '#![cfg(test)]' in clean_line:
            pending_test = True
            brace_stack = [True]

        # Check for test attributes on this item
        if '#[cfg(test)]' in clean_line or '#[test]' in clean_line or '#[bench]' in clean_line:
            pending_test = True

        # Check for test module declarations
        if re.search(r'\bmod\s+(?:tests?|testing|bench(?:marks?)?)\b', clean_line):
            pending_test = True

        is_curr_test = pending_test or (len(brace_stack) > 0 and brace_stack[-1])

        # Track braces in clean_line
        for ch in clean_line:
            if ch == '{':
                brace_stack.append(is_curr_test)
                pending_test = False
            elif ch == '}':
                if brace_stack:
                    brace_stack.pop()

        if clean_line.endswith(';') and not ('{' in clean_line):
            pending_test = False

        in_test = (len(brace_stack) > 0 and brace_stack[-1]) or pending_test

        docs = [t for (c_idx, t) in recent_doc_comments if idx - c_idx <= 20]
        comms = [t for (c_idx, t) in recent_comments if idx - c_idx <= 20]

        yield {
            'line_idx': idx,
            'line_num': line_num,
            'raw_line': line,
            'clean_line': clean_line,
            'in_test': in_test,
            'doc_comments': docs,
            'comments_above': comms,
        }

        # Clear recent comments once an item declaration is seen
        if clean_line and not clean_line.startswith('#'):
            recent_comments = []
            recent_doc_comments = []
