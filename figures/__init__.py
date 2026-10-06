"""Publication figure helpers: shared cube/CHM utilities and plot-only entry points."""

from figures.overlays import DDPM_SCATTER, MODEL_METRICS, SR3_LINE

__all__ = [
    "DDPM_SCATTER",
    "MODEL_METRICS",
    "SR3_LINE",
    "plot_teaser",
]


def __getattr__(name: str):
    if name == "plot_teaser":
        from figures.plot_teaser import plot_teaser

        return plot_teaser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
