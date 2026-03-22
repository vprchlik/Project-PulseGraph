"""FastAPI application for PulseGraph."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pulsegraph import __version__
from pulsegraph.api.state import AppState

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize state on startup, clean up on shutdown."""
    logger.info("PulseGraph API starting up")
    app.state.pg = AppState()
    app.state.pg.load()
    yield
    logger.info("PulseGraph API shutting down")


app = FastAPI(
    title="PulseGraph API",
    description="Future search engine for public entities",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pulsegraph.api.routes import router  # noqa: E402

app.include_router(router)
