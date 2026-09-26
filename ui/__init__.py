"""Gradio front-end. The only package allowed to import gradio; `src/` stays
UI-free so generation and measurement remain testable without a browser."""
from .layout import build_ui

__all__ = ["build_ui"]
