# -*- coding: utf-8 -*-
"""
Visual theme for the dashboard: palette constants, the global CSS Streamlit
doesn't cover, and small helpers for the boxed layout every panel uses.

Colour scheme is a light beige/cream ground with navy as the primary ink and a
deep magenta as the accent, set in Times New Roman. The palette values here
mirror .streamlit/config.toml — that file themes Streamlit's own widgets, this
module themes everything we draw ourselves (cards, Plotly figures). Change a
colour in both places or the two halves drift apart.
"""

import streamlit as st

# --- Palette ---------------------------------------------------------------

CREAM = "#FAF5E6"        # page background
CARD_BG = "#FFFCF3"      # panel/card fill, a shade lighter than the page
CARD_BG_ALT = "#F6EEDA"  # secondary fill for nested/table rows
BORDER = "#DFD2AE"       # warm tan rule, visible on both fills
BORDER_STRONG = "#C9B888"

NAVY = "#1B2A4A"         # primary ink and headings
NAVY_SOFT = "#2E4372"    # secondary ink
MAGENTA = "#7B2D5E"      # accent: section rules, key figures, active states
MAGENTA_SOFT = "#9C4478"
MUTED = "#6B6350"        # captions, units, helper text

SERIF_STACK = '"Times New Roman", Times, Georgia, "Nimbus Roman", serif'


def _css() -> str:
    return f"""
    <style>
    /* Typography ------------------------------------------------------- */
    /* Icon spans are excluded rather than overridden: Streamlit paints its
       chrome icons as Material Symbols *ligatures* through its own hashed
       class, so any rule of ours that wins on that span turns the glyph into
       its literal name ("keyboard_double_arrow_left"). Leaving those spans
       alone lets Streamlit's own font rule apply untouched. */
    html, body, [class*="css"], .stApp,
    .stApp p, .stApp li, .stApp label, .stApp div,
    .stApp span:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]),
    .stMarkdown, button, input, select, textarea {{
        font-family: {SERIF_STACK} !important;
    }}
    .stApp {{
        background-color: {CREAM};
        color: {NAVY};
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        font-family: {SERIF_STACK} !important;
        color: {NAVY};
        letter-spacing: 0.01em;
    }}
    .stApp h1 {{
        font-size: 2.15rem;
        font-weight: 700;
        border-bottom: 2px solid {MAGENTA};
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }}
    .stApp h2 {{ font-size: 1.5rem; }}
    .stApp h3 {{ font-size: 1.2rem; }}

    /* Streamlit draws its chrome icons (sidebar collapse, expander arrows,
       alert glyphs) as Material Symbols *ligatures* — the glyph only appears
       because the icon font maps the literal text. The blanket serif rule
       above would otherwise render them as the words themselves, e.g. a
       stray "keyboard_double_arrow_left" above the sidebar, so the icon font
       is restored here at higher specificity. */
       The :not() exclusions in the rule above keep our serif off them; this
       is the belt-and-braces fallback in case a future Streamlit build stops
       setting the family itself. */
    .stApp span[data-testid="stIconMaterial"],
    section[data-testid="stSidebar"] span[data-testid="stIconMaterial"] {{
        font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                     "Material Icons";
    }}

    /* Code/monospace should stay monospace rather than inherit the serif. */
    .stApp code, .stApp pre, .stApp kbd, .stApp samp,
    .stApp [data-testid="stCode"] {{
        font-family: "DejaVu Sans Mono", "Consolas", "Courier New", monospace !important;
    }}

    /* Keep the content column a comfortable reading width on a laptop
       instead of letting it stretch edge to edge on a wide display. */
    .block-container {{
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }}

    /* Sidebar ---------------------------------------------------------- */
    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG_ALT};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] label {{
        color: {NAVY} !important;
    }}

    /* Streamlit's selectbox is a searchable combobox that keeps text-input
       focus after a selection, which otherwise leaves a blinking text cursor
       sitting in the sidebar. */
    [data-testid="stSelectbox"] input {{
        caret-color: transparent !important;
    }}

    /* Cards ------------------------------------------------------------ */
    .aqi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 1.05rem 1.25rem 1.15rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(27, 42, 74, 0.07);
    }}
    .aqi-card-title {{
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: {MAGENTA};
        margin: 0 0 0.75rem 0;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid {BORDER};
    }}

    /* Reading grid: label / value+unit rows inside a card --------------- */
    .reading-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(196px, 1fr));
        gap: 0.55rem 1.4rem;
    }}
    .reading {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.6rem;
        padding: 0.42rem 0.6rem;
        background: {CARD_BG_ALT};
        border-left: 3px solid {NAVY_SOFT};
        border-radius: 3px;
    }}
    .reading-label {{
        color: {NAVY};
        font-size: 0.95rem;
    }}
    .reading-value {{
        color: {MAGENTA};
        font-size: 1.06rem;
        font-weight: 700;
        white-space: nowrap;
    }}
    .reading-unit {{
        color: {MUTED};
        font-size: 0.82rem;
        font-weight: 400;
        margin-left: 0.15rem;
    }}

    /* Hero AQI box ----------------------------------------------------- */
    .aqi-hero {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1.2rem;
        border-radius: 8px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1rem;
        color: #FFFFFF;
        box-shadow: 0 2px 6px rgba(27, 42, 74, 0.16);
    }}
    .aqi-hero-value {{
        font-size: 3.1rem;
        font-weight: 700;
        line-height: 1;
    }}
    .aqi-hero-label {{
        font-size: 0.76rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        opacity: 0.9;
    }}
    .aqi-hero-category {{
        font-size: 1.35rem;
        font-weight: 700;
    }}
    .aqi-hero-meta {{
        font-size: 0.9rem;
        opacity: 0.93;
        text-align: right;
    }}

    /* Divider + captions ----------------------------------------------- */
    hr, [data-testid="stDivider"] {{ border-color: {BORDER} !important; }}
    .aqi-caption {{
        color: {MUTED};
        font-size: 0.88rem;
        font-style: italic;
        margin: 0.15rem 0 0.9rem 0;
    }}
    </style>
    """


def apply_theme() -> None:
    """Injects the global stylesheet. Call once, right after set_page_config."""
    st.markdown(_css(), unsafe_allow_html=True)


def card(title: str, body_html: str) -> str:
    """A titled panel. Returns HTML for st.markdown(..., unsafe_allow_html=True)."""
    return (
        f'<div class="aqi-card">'
        f'<div class="aqi-card-title">{title}</div>'
        f"{body_html}"
        f"</div>"
    )


def reading(label: str, value: str, unit: str = "") -> str:
    """One label/value row for use inside a reading_grid()."""
    unit_html = f'<span class="reading-unit">{unit}</span>' if unit else ""
    return (
        f'<div class="reading">'
        f'<span class="reading-label">{label}</span>'
        f'<span class="reading-value">{value}{unit_html}</span>'
        f"</div>"
    )


def reading_grid(rows: list) -> str:
    return f'<div class="reading-grid">{"".join(rows)}</div>'


def style_figure(fig, height: int = 320):
    """Puts a Plotly figure on the same cream/navy/serif footing as the CSS."""
    fig.update_layout(
        height=height,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family=SERIF_STACK, color=NAVY, size=13),
        title=dict(font=dict(family=SERIF_STACK, color=NAVY, size=17)),
        margin=dict(l=55, r=25, t=55, b=45),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER_STRONG, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER_STRONG, zerolinecolor=BORDER),
    )
    return fig
