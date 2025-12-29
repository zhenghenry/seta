#!/usr/bin/env python3
"""
Wrapper to fix Python 3.12+ asyncio compatibility before running Harbor.

Python 3.12+ removed the child watcher API. This wrapper patches asyncio
to provide a working implementation before Harbor starts.
"""
import sys
import os
import asyncio

# Fix for Python 3.12+ asyncio child watcher issue
if sys.version_info >= (3, 12) and sys.platform != 'win32':
    import asyncio.events
    import asyncio.unix_events

    _watcher = asyncio.unix_events.ThreadedChildWatcher()
    _attached_loop = None

    def _ensure_attached() -> None:
        global _attached_loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop is not _attached_loop:
            try:
                _watcher.attach_loop(loop)
                _attached_loop = loop
            except Exception:
                pass

    def _patched_get_child_watcher():
        _ensure_attached()
        return _watcher

    def _patched_set_child_watcher(watcher):
        # Allow libraries to replace the watcher if they insist.
        global _watcher
        global _attached_loop
        _watcher = watcher
        _attached_loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and hasattr(_watcher, "attach_loop"):
            try:
                _watcher.attach_loop(loop)
                _attached_loop = loop
            except Exception:
                pass

    asyncio.events.get_child_watcher = _patched_get_child_watcher
    asyncio.events.set_child_watcher = _patched_set_child_watcher

    asyncio.get_child_watcher = _patched_get_child_watcher
    asyncio.set_child_watcher = _patched_set_child_watcher

from harbor.cli.main import app

if __name__ == "__main__":
    exit_code: int = 0
    exc: BaseException | None = None
    try:
        app()
    except SystemExit as e:
        code = e.code
        if code is None:
            exit_code = 0
        elif isinstance(code, int):
            exit_code = code
        else:
            exit_code = 1
        exc = e
    except BaseException as e:
        exit_code = 1
        exc = e
    finally:
        if sys.version_info >= (3, 12) and sys.platform != 'win32':
            try:
                watcher = asyncio.events.get_child_watcher()
                close = getattr(watcher, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass

        if os.getenv("HARBOR_FORCE_EXIT", "0") == "1":
            os._exit(exit_code)

    if exc is not None:
        raise exc
