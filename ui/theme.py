"""Neutral, dense theme: zinc greys, one blue accent, system fonts (no web-font
fetch, so it renders the same offline and on Spaces)."""
from __future__ import annotations

from pathlib import Path

import gradio as gr

CSS = (Path(__file__).with_name("style.css")).read_text(encoding="utf-8")

# Theme tokens reused across light/dark variables.
NEUTRAL_900 = "*neutral_900"
NEUTRAL_700 = "*neutral_700"
PRIMARY_600 = "*primary_600"


def build_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.zinc,
        text_size=gr.themes.sizes.text_sm,
        spacing_size=gr.themes.sizes.spacing_sm,
        radius_size=gr.themes.sizes.radius_sm,
        font=["Helvetica Neue", "Segoe UI", "Roboto", "Arial", "sans-serif"],
        font_mono=["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"],
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        background_fill_primary="white",
        background_fill_primary_dark=NEUTRAL_900,
        background_fill_secondary="*neutral_50",
        background_fill_secondary_dark="*neutral_950",
        block_background_fill="white",
        block_background_fill_dark=NEUTRAL_900,
        block_border_width="1px",
        block_shadow="none",
        block_shadow_dark="none",
        # Image/gallery labels float over the content; keep them opaque.
        block_label_background_fill="white",
        block_label_background_fill_dark=NEUTRAL_900,
        block_label_text_color="*neutral_500",
        block_label_text_color_dark="*neutral_400",
        block_label_border_width="1px",
        block_title_text_weight="600",
        button_primary_background_fill=NEUTRAL_900,
        button_primary_background_fill_hover=NEUTRAL_700,
        button_primary_text_color="white",
        button_primary_background_fill_dark="*neutral_100",
        button_primary_background_fill_hover_dark="*neutral_300",
        button_primary_text_color_dark=NEUTRAL_900,
        button_secondary_background_fill="white",
        button_secondary_background_fill_dark="*neutral_800",
        button_secondary_border_color="*neutral_300",
        button_secondary_border_color_dark=NEUTRAL_700,
        button_secondary_background_fill_hover="*neutral_100",
        button_secondary_background_fill_hover_dark=NEUTRAL_700,
        slider_color=PRIMARY_600,
        slider_color_dark="*primary_500",
        color_accent=PRIMARY_600,
        link_text_color=PRIMARY_600,
        link_text_color_dark="*primary_400",
        shadow_drop="none",
    )
