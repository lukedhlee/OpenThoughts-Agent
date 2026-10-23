"""Copy query-optimize and optionally accelerate its untimed reference check.

The first test validates the fixed supplied slow query against the golden query.
A temporary indexed copy of senses gives that exact query fast lookups, without
changing main database contents or any assertion. The timed solution comparison
opens a fresh connection and retains its original database, query, and threshold.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ANCHOR = '''        golden_results = _fetch_all(con, q_golden)
        my_sql_results = _fetch_all(con, q_my_sql_query)'''
REPLACEMENT = '''        golden_results = _fetch_all(con, q_golden)
        # Index only the supplied reference query's untimed correctness check.
        # The timed solution tests open fresh connections to the unchanged main DB.
        con.execute("CREATE TEMP TABLE senses AS SELECT * FROM main.senses")
        con.execute("CREATE INDEX temp.cf2_senses_wordid_synsetid ON senses(wordid, synsetid)")
        my_sql_results = _fetch_all(con, q_my_sql_query)'''


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--reference-index', action='store_true')
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    assert source.name == destination.name == 'query-optimize'
    assert not destination.exists(), 'Refuse to overwrite a task copy'
    before = hashes(source)
    shutil.copytree(source, destination, symlinks=True)
    assert hashes(destination) == before
    if args.reference_index:
        test = destination / 'tests/test_outputs.py'
        original = test.read_text()
        assert original.count(ANCHOR) == 1, 'Reference-check source drifted'
        repaired = original.replace(ANCHOR, REPLACEMENT)
        trees = [ast.parse(text) for text in (original, repaired)]
        assertions = [[ast.dump(node) for node in ast.walk(tree)
                       if isinstance(node, ast.Assert)] for tree in trees]
        assert assertions[0] == assertions[1], 'All correctness predicates must remain identical'
        for name in ('test_compare_golden_vs_solution_runtime', 'test_outputs_match_exactly',
                     'test_check_for_db_modifications', 'test_solution_contains_single_sql_query',
                     'test_solution_is_small'):
            functions = [next(node for node in tree.body
                              if isinstance(node, ast.FunctionDef) and node.name == name)
                         for tree in trees]
            assert ast.dump(functions[0]) == ast.dump(functions[1]), name
        test.write_text(repaired)
    after = hashes(destination)
    changed = sorted(k for k in before if before[k] != after[k])
    assert changed == (['tests/test_outputs.py'] if args.reference_index else [])
    assert set(before) == set(after)
    assert hashes(source) == before, 'Source task must remain immutable'
    print(json.dumps(dict(source=str(source), destination=str(destination),
                          reference_index=args.reference_index, changed=changed,
                          original_sha256=before, repaired_sha256=after), indent=2))


if __name__ == '__main__':
    main()
