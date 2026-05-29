"""About / Who We Are page.

Register in your navigation, e.g.:

    about_page = st.Page("about.py", title="Who We Are", icon=":material/groups:", url_path="about")
    pg = st.navigation([..., about_page])
    pg.run()

Place member photos in assets/team/ (square images work best, e.g. 400x400).
Missing files fall back to a generated initials avatar, so the layout never breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).parent / "assets" / "team"


@dataclass(frozen=True)
class Member:
    name: str
    role: str
    bio: str
    photo: str  # filename inside ASSETS_DIR
    links: dict[str, str] = field(default_factory=dict)  # label -> url


# --- Edit your team here ----------------------------------------------------
MISSION = (
    "We build AI-powered skill-editing tools that let business users define, "
    "test, and ship agent skills without touching code."
)

PILLARS: list[tuple[str, str, str]] = [
    (":material/target:", "Our Mission", "Make agentic AI configurable by the people who use it."),
    (":material/build:", "What We Do", "Schema-driven skill editing, validation, and live testing."),
    (":material/bolt:", "How We Work", "Explicit and structured. No silent fallbacks."),
]

MEMBERS: list[Member] = [
    Member(
        name="Riadh",
        role="Founder / Quant & AI Engineering",
        bio="15+ years in quantitative finance, now building production agentic systems.",
        photo="riadh.jpg",
        links={"LinkedIn": "https://www.linkedin.com/"},
    ),
    Member(
        name="Member Two",
        role="Role",
        bio="Short bio goes here.",
        photo="member_two.jpg",
    ),
    Member(
        name="Member Three",
        role="Role",
        bio="Short bio goes here.",
        photo="member_three.jpg",
    ),
]
# ---------------------------------------------------------------------------


def _initials_avatar(name: str) -> str:
    """Return a data-URI SVG avatar with the member's initials as a fallback."""
    initials = "".join(part[0].upper() for part in name.split()[:2]) or "?"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">'
        f'<rect width="200" height="200" fill="#4F46E5"/>'
        f'<text x="50%" y="50%" dy=".35em" text-anchor="middle" '
        f'font-family="sans-serif" font-size="80" fill="white">{initials}</text>'
        f"</svg>"
    )
    import base64

    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"


def _photo_src(member: Member) -> str:
    path = ASSETS_DIR / member.photo
    return str(path) if path.exists() else _initials_avatar(member.name)


_CSS = """
<style>
.team-card {
    background: var(--secondary-background-color, #f7f7f9);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    height: 100%;
    text-align: center;
    transition: transform .15s ease, box-shadow .15s ease;
}
.team-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.10);
}
.team-card img {
    width: 116px; height: 116px;
    object-fit: cover;
    border-radius: 50%;
    margin: 0 auto .8rem;
    display: block;
    border: 3px solid rgba(79,70,229,0.35);
}
.team-card .name { font-weight: 700; font-size: 1.08rem; margin-bottom: .15rem; }
.team-card .role { color: #4F46E5; font-size: .85rem; font-weight: 600;
                   text-transform: uppercase; letter-spacing: .03em; margin-bottom: .6rem; }
.team-card .bio  { font-size: .9rem; opacity: .85; line-height: 1.45; }
.team-card .links { margin-top: .7rem; font-size: .85rem; }
.team-card .links a { color: #4F46E5; text-decoration: none; margin: 0 .4rem; }
.team-card .links a:hover { text-decoration: underline; }
</style>
"""


def _card_html(member: Member) -> str:
    links = ""
    if member.links:
        anchors = " · ".join(
            f'<a href="{url}" target="_blank">{label}</a>'
            for label, url in member.links.items()
        )
        links = f'<div class="links">{anchors}</div>'
    return (
        f'<div class="team-card">'
        f'<img src="{_photo_src(member)}" alt="{member.name}"/>'
        f'<div class="name">{member.name}</div>'
        f'<div class="role">{member.role}</div>'
        f'<div class="bio">{member.bio}</div>'
        f"{links}"
        f"</div>"
    )


def render_about() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    st.title("Who We Are")
    st.markdown(f"#### {MISSION}")
    st.write("")

    cols = st.columns(len(PILLARS), gap="large")
    for col, (icon, title, text) in zip(cols, PILLARS):
        with col:
            st.subheader(f"{icon} {title}")
            st.write(text)

    st.divider()
    st.subheader("The Team")
    st.write("")

    cols = st.columns(len(MEMBERS), gap="large")
    for col, member in zip(cols, MEMBERS):
        with col:
            st.markdown(_card_html(member), unsafe_allow_html=True)


# Allows the file to be used directly as an st.Page target.
render_about()
