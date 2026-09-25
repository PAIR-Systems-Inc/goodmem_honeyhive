"""HoneyHive tracing that can never run a GoodMem operation twice.

HoneyHive's ``@trace`` wraps the traced function *and* its own bookkeeping in
one ``try``. When the exception it catches contains the text ``Tracer error``
it assumes tracing failed and calls the function again, untraced, with the
same arguments (``honeyhive/tracer/instrumentation/decorators.py``). Our
methods raise errors that carry the GoodMem server's message, and the server
can echo request content, so a failed ``create_memory`` or ``delete_memory``
whose error text contained that phrase was sent a second time.

:func:`traced` puts a run-once guard between ``@trace`` and the method: within
one call, a second invocation returns the first invocation's result or
re-raises its exception instead of executing again. Tracing is unchanged -- the
same span name, event type and captured inputs.
"""

from __future__ import annotations

import contextvars
import functools
from typing import Any, Callable, TypeVar, cast

from honeyhive import trace

F = TypeVar("F", bound=Callable[..., Any])

_UNSET = object()


def traced(*, event_type: str, event_name: str) -> Callable[[F], F]:
    """Like ``honeyhive.trace``, but the method body runs at most once per call."""

    def decorate(method: F) -> F:
        # One slot per call. A dict is used rather than setting the variable
        # inside the body, so the outcome is visible even if HoneyHive ever
        # runs the retry in a copied context.
        attempt: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
            f"goodmem_{method.__name__}_attempt", default=None
        )

        @functools.wraps(method)
        def once(*args: Any, **kwargs: Any) -> Any:
            slot = attempt.get()
            if slot is None:  # called without the outer wrapper: just run
                return method(*args, **kwargs)
            if slot["error"] is not None:
                raise slot["error"]
            if slot["result"] is not _UNSET:
                return slot["result"]
            try:
                slot["result"] = method(*args, **kwargs)
            except BaseException as error:
                slot["error"] = error
                raise
            return slot["result"]

        traced_once = trace(event_type=event_type, event_name=event_name)(once)

        @functools.wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            token = attempt.set({"result": _UNSET, "error": None})
            try:
                return traced_once(*args, **kwargs)
            finally:
                attempt.reset(token)

        return cast(F, call)

    return decorate
