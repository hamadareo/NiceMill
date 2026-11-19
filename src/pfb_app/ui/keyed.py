"""Keyed containers that never leave old copies on the screen.

Streamlit's frontend uses a block's id as its React key, and a keyed container's id comes from
its key alone, not from where it is. When a keyed container moves up among its siblings from
one run to the next (a one-time message or the approval stamp above it is gone), the new block
and the old one, which is only cleared when the run ends, are siblings with the same React key
for a moment. React then loses track of the old block and its DOM stays for good: greyed, with
any tooltip that was open, over the next screens too (step 5's buttons on step 6, owner's report
2026-10-24; scripts/browser_stale_check.py shows it in a browser).

So a keyed block is always made here, as the only child of an unkeyed container: the outer
container's React key is its position, and the keyed one is always its first and only child.
The key's CSS class (`st-key-<key>`) is on the inner block, as before.
"""
import streamlit as st
from streamlit.delta_generator import DeltaGenerator


def container(key: str, **kwargs) -> DeltaGenerator:
    """`st.container(key=key, **kwargs)`, alone in an unkeyed container."""
    return st.container().container(key=key, **kwargs)


def expander(label: str, *, key: str, **kwargs):
    """`st.expander(label, key=key, **kwargs)`, alone in an unkeyed container."""
    return st.container().expander(label, key=key, **kwargs)
