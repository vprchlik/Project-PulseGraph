"""SQLAlchemy models for PulseGraph's relational schema."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from pulsegraph.config import settings


class Base(DeclarativeBase):
    pass


class Repository(Base):
    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, default="")
    language = Column(String(50), default="")
    license = Column(String(50), default="")
    created_at = Column(DateTime, nullable=True)
    topics = Column(ARRAY(String), default=[])
    org = Column(String(100), default="")
    stratum = Column(String(20), default="unknown")
    archived = Column(Boolean, default=False)
    is_fork = Column(Boolean, default=False)
    last_updated = Column(DateTime, default=datetime.utcnow)

    signals = relationship("DailySignal", back_populates="repository")
    events = relationship("RepoEvent", back_populates="repository")


class DailySignal(Base):
    __tablename__ = "daily_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    event_date = Column(Date, nullable=False)
    stars = Column(Integer, default=0)
    forks = Column(Integer, default=0)
    pushes = Column(Integer, default=0)
    issues = Column(Integer, default=0)
    pull_requests = Column(Integer, default=0)
    releases = Column(Integer, default=0)

    repository = relationship("Repository", back_populates="signals")

    __table_args__ = (
        Index("ix_daily_signals_repo_date", "repo_id", "event_date", unique=True),
    )


class RepoEvent(Base):
    __tablename__ = "repo_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    event_type = Column(String(50), nullable=False)
    event_date = Column(Date, nullable=False)
    title = Column(Text, default="")
    body = Column(Text, default="")
    url = Column(Text, default="")
    source = Column(String(50), default="")
    summary = Column(Text, default="")
    metadata_ = Column("metadata", JSONB, default={})

    repository = relationship("Repository", back_populates="events")

    __table_args__ = (
        Index("ix_repo_events_repo_date", "repo_id", "event_date"),
    )


class EntityEdge(Base):
    __tablename__ = "entity_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_repo_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    target_repo_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    edge_type = Column(String(50), nullable=False)
    weight = Column(Float, default=1.0)
    metadata_ = Column("metadata", JSONB, default={})

    __table_args__ = (
        Index("ix_entity_edges_source", "source_repo_id", "edge_type"),
        Index("ix_entity_edges_target", "target_repo_id", "edge_type"),
    )


class ForecastResult(Base):
    __tablename__ = "forecast_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    forecast_date = Column(Date, nullable=False)
    horizon = Column(Integer, nullable=False)
    signal_name = Column(String(50), default="stars")
    model_name = Column(String(100), nullable=False)
    regime_probs = Column(JSONB, default={})
    median_forecast = Column(ARRAY(Float), default=[])
    lower_80 = Column(ARRAY(Float), default=[])
    upper_80 = Column(ARRAY(Float), default=[])
    lower_95 = Column(ARRAY(Float), default=[])
    upper_95 = Column(ARRAY(Float), default=[])
    confidence_score = Column(Float, default=0.0)
    metadata_ = Column("metadata", JSONB, default={})

    __table_args__ = (
        Index("ix_forecast_repo_date_horizon", "repo_id", "forecast_date", "horizon"),
    )


def get_engine(url: str | None = None):
    return create_engine(url or settings.database_url)


def get_session(url: str | None = None) -> Session:
    engine = get_engine(url)
    return sessionmaker(bind=engine)()


def create_all_tables(url: str | None = None):
    engine = get_engine(url)
    Base.metadata.create_all(engine)
