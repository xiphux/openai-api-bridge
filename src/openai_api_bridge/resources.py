"""The process-wide resource graph, as one typed object.

The lifespan in ``main`` builds these in a fixed order and every route needs
some of them. They used to be set as eight loose attributes on ``app.state``
and read back with a declared type at ~20 call sites:

    dispatcher: BackendDispatcher = request.app.state.dispatcher

``app.state`` is untyped, so that annotation was an assertion the type checker
accepted without evidence — a renamed attribute, or a route reached before the
lifespan ran, produced an ``AttributeError`` and a 500 rather than anything
mypy could have caught. Twenty repetitions of an unchecked cast is also twenty
places to update when the graph changes.

One frozen container instead, installed once and fetched through
:func:`resources`. The cast happens in exactly one place, the field names are
checked everywhere else, and what the bridge holds for the life of the process
is written down in one list rather than inferred from eight assignments.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, Request

from .config import BridgeSettings, ProvidersFile
from .dispatcher import BackendDispatcher
from .infra.db import Database
from .infra.eviction import EvictionLoop
from .infra.filestore import FileStore
from .infra.jobstore import JobStore
from .infra.tasks import TaskScheduler

# Deliberately not "state" or one of the field names: it occupies a single slot
# on Starlette's shared State object, which middleware and extensions also
# write to.
_ATTR = "bridge_resources"


@dataclass(frozen=True, slots=True)
class BridgeResources:
    """Everything built at startup and torn down at shutdown.

    Frozen because nothing should be swapping a live dispatcher or file store
    out from under an in-flight request; the lifespan builds the graph once.

    ``providers``, ``db`` and ``eviction`` aren't read by any route today. They
    stay because this type is the inventory of what the process holds, and a
    list that only covers what routes happen to touch is a worse answer to
    "what is running here".
    """

    settings: BridgeSettings
    providers: ProvidersFile
    db: Database
    filestore: FileStore
    jobstore: JobStore
    dispatcher: BackendDispatcher
    scheduler: TaskScheduler
    eviction: EvictionLoop


def install(app: FastAPI, resources: BridgeResources) -> None:
    """Attach the resource graph to the app. Called once, from the lifespan."""
    setattr(app.state, _ATTR, resources)


def resources(request: Request) -> BridgeResources:
    """The resource graph for the app serving this request.

    The one place ``app.state``'s untyped value is trusted, so it's also the
    one place that can explain itself when it isn't there — which means the
    lifespan didn't run. That happens with a bare ``TestClient(app)`` used
    without its context manager, where the old failure was an ``AttributeError``
    naming a single attribute and pointing at the route rather than the cause.
    """
    try:
        found: BridgeResources = getattr(request.app.state, _ATTR)
    except AttributeError:
        raise RuntimeError(
            "Bridge resources are not installed on this app — its lifespan has "
            "not run. Enter the app's lifespan context (e.g. `with TestClient(app):`) "
            "before serving requests."
        ) from None
    return found
