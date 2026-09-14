"""Introspect the transcribe_cpp python API surface."""
import inspect
import sys

import transcribe_cpp as t

for cls in ["Session", "Model", "Capabilities", "Feature", "Task", "Stream", "Result", "Token", "Sequence", "Timings"]:
    c = getattr(t, cls, None)
    if c is None:
        continue
    print("=" * 60)
    print(cls)
    print("  doc:", (inspect.getdoc(c) or "")[:400].replace("\n", " | "))
    try:
        print("  __init__", inspect.signature(c.__init__))
    except Exception as e:
        print("  __init__ err", e)
    for n in sorted(dir(c)):
        if n.startswith("_"):
            continue
        try:
            a = inspect.getattr_static(c, n)
        except Exception:
            continue
        if callable(a):
            try:
                print("    -", n, inspect.signature(getattr(c, n)))
            except Exception:
                print("    -", n, "(?)")
        else:
            print("    .", n, "=", repr(a)[:100])
print("=" * 60)
print("module transcribe():", inspect.signature(t.transcribe) if callable(getattr(t, "transcribe", None)) else t.transcribe)
