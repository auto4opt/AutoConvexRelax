"""Tripartite graph construction and graph encoding."""

from .conversion import qcqp_to_heterodata
from .encoder import GNNEncoder

__all__ = ["GNNEncoder", "qcqp_to_heterodata"]
