"""Tok potrubí. Zná jen porty z `atr.ports`, žádný konkrétní nástroj."""

from atr.pipeline.pipeline import Pipeline, PipelineError

__all__ = ["Pipeline", "PipelineError"]
