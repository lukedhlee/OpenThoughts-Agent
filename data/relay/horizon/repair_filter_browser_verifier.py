"""Copy filter-js-from-html and optionally repair browser lifecycle and loading.

Killing chromedriver before driver.quit() strands the browser processes. Across
28 batches they exhaust sandbox memory, preventing the verifier from inspecting
pages. Reorder cleanup while retaining its corpus, browser options, alert wait,
20-second per-page intent, assertions, and overall budget. Selenium gets a
15-second page-load deadline, leaving the original three-second alert wait. A
resource-load timeout still requires an inspectable DOM and a real alert check.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ANCHOR = '''                # Force kill Chrome if it's still running
                driver.service.process.kill() if hasattr(driver, "service") else None
                driver.quit()
                print("  Driver closed")
            except Exception as e:
                print(f"  Error closing driver: {e}")
                # Try to kill any remaining Chrome processes
                try:
                    subprocess.run(["pkill", "-f", "chrome"], timeout=2)
                except Exception:
                    pass'''
REPLACEMENT = '''                # Close the browser before terminating the service that controls it.
                driver.quit()
                print("  Driver closed")
            except Exception as e:
                print(f"  Error closing driver: {e}")
                if hasattr(driver, "service") and driver.service.process:
                    driver.service.process.kill()'''


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--repair-browser', action='store_true')
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    assert source.name == destination.name == 'filter-js-from-html'
    assert not destination.exists(), 'Refuse to overwrite a task copy'
    before = hashes(source)
    shutil.copytree(source, destination, symlinks=True)
    assert hashes(destination) == before
    if args.repair_browser:
        test = destination / 'tests/test_outputs.py'
        original = test.read_text()
        assert original.count(ANCHOR) == 1, 'Browser-cleanup source drifted'
        repaired = original.replace(ANCHOR, REPLACEMENT)
        create = '        driver = webdriver.Chrome(options=chrome_options)\n'
        assert repaired.count(create) == 1
        repaired = repaired.replace(create, create +
                                    '        driver.set_page_load_timeout(max(1, timeout - 5))\n')
        load = '''        except Exception as e:
            print(f"  Failed to load URL: {e}")
            return False
'''
        bounded = '''        except TimeoutException:
            # Resource downloads may stall after the document is ready. Still
            # inspect the actual browser for alerts instead of accepting a timeout.
            state = driver.execute_script("return document.readyState")
            if state not in ("interactive", "complete"):
                raise RuntimeError(f"Page did not produce an inspectable DOM: {state}")
            print("  Page resources pending; inspecting loaded DOM for alerts")
        except Exception as e:
            print(f"  Failed to load URL: {e}")
            return False
'''
        assert repaired.count(load) == 1
        repaired = repaired.replace(load, bounded)
        assertions = [[ast.dump(node) for node in ast.walk(ast.parse(text))
                       if isinstance(node, ast.Assert)] for text in (original, repaired)]
        assert assertions[0] == assertions[1], 'All original assertions must remain identical'
        original_functions = {node.name: ast.dump(node) for node in ast.parse(original).body
                              if isinstance(node, ast.FunctionDef)}
        repaired_functions = {node.name: ast.dump(node) for node in ast.parse(repaired).body
                              if isinstance(node, ast.FunctionDef)}
        assert {name for name in original_functions
                if original_functions[name] != repaired_functions[name]} == {'run_test_alert_file'}
        test.write_text(repaired)
    after = hashes(destination)
    changed = sorted(k for k in before if before[k] != after[k])
    assert changed == (['tests/test_outputs.py'] if args.repair_browser else [])
    assert set(before) == set(after)
    assert hashes(source) == before
    print(json.dumps(dict(source=str(source), destination=str(destination),
                          repair_browser=args.repair_browser, changed=changed,
                          original_sha256=before, repaired_sha256=after), indent=2))


if __name__ == '__main__':
    main()
