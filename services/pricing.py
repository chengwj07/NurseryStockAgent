"""Per-tree price formulas (power laws of crown width W and, optionally, height H)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional


CM_PER_M: float = 100.0  # the price formulas use centimetres


class PricingModel:
    """Species table with price formulas.

    Each supported species has two formulas of the form
    ``V = coef * H**h_exp * W**w_exp`` (``h_exp`` optional):

    * ``single``: only the original image is available (uses W only);
    * ``dem``: original image + DEM available (may use H as well).

    ``W`` (equivalent crown diameter) and ``H`` (tree height) are in
    **centimetres** inside the formulas. The public API takes **metres** (the
    unit used everywhere else in the app) and converts at the formula boundary.
    Species without ``formulas`` have no evaluation formula.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the model.

        Args:
            config: Parsed content of ``price_table.json``.
        """
        self._currency: str = str(config.get("currency", "CNY"))
        self._species: dict[str, dict[str, Any]] = dict(config.get("species", {}))
        self._alias: dict[str, str] = {}
        for name, info in self._species.items():
            self._alias[name.lower()] = name
            for alias in info.get("aliases", []):
                self._alias[str(alias).lower()] = name

    @classmethod
    def from_json(cls, path: Path) -> "PricingModel":
        """Load a pricing model from a JSON file.

        Args:
            path: Path of ``price_table.json``.

        Returns:
            The loaded model.
        """
        with path.open("r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @property
    def currency(self) -> str:
        """Currency unit label."""
        return self._currency

    def resolve(self, name: str) -> Optional[str]:
        """Map a user supplied name or alias to the canonical species name.

        Args:
            name: Species name as typed by the user.

        Returns:
            Canonical name, or ``None`` if the species is not in the table.
        """
        return self._alias.get(name.strip().lower())

    def prompt_for(self, name: str) -> str:
        """Return the text prompt sent to SAM3 for a species.

        Args:
            name: Species name as typed by the user.

        Returns:
            The configured prompt, or ``name`` itself if unknown.
        """
        canonical = self.resolve(name)
        if canonical is None:
            return name.strip()
        return str(self._species[canonical].get("prompt", canonical))

    def has_formula(self, name: str) -> bool:
        """Whether a species has an evaluation formula.

        Args:
            name: Species name or alias.

        Returns:
            ``True`` if the species can be valued.
        """
        canonical = self.resolve(name)
        return bool(canonical and self._species[canonical].get("formulas"))

    def supported_species(self) -> list[str]:
        """Names of all species that have a formula."""
        return [n for n, info in self._species.items() if info.get("formulas")]

    def _formula(self, species: str, use_dem: bool) -> Optional[dict[str, float]]:
        """Return the formula dict for a species and mode, if any."""
        canonical = self.resolve(species)
        if canonical is None:
            return None
        formulas = self._species[canonical].get("formulas") or {}
        return formulas.get("dem" if use_dem else "single")

    def tree_price(
        self,
        species: str,
        crown_w: float,
        height: Optional[float] = None,
        use_dem: bool = False,
    ) -> Optional[float]:
        """Price of one tree.

        Args:
            species: Species name or alias.
            crown_w: Equivalent crown diameter W in **metres**.
            height: Tree height H in **metres** (needed only if the formula uses H).
            use_dem: Use the original-image + DEM formula.

        Returns:
            The price, or ``None`` if there is no formula or an input is invalid.
        """
        formula = self._formula(species, use_dem)
        if formula is None or crown_w is None or not math.isfinite(crown_w) or crown_w <= 0:
            return None
        coef = float(formula["coef"]) if "coef" in formula else math.exp(float(formula["ln_coef"]))
        value = coef * (crown_w * CM_PER_M) ** float(formula["w_exp"])  # W in cm
        if "h_exp" in formula:
            if height is None or not math.isfinite(height) or height <= 0:
                return None
            value *= (height * CM_PER_M) ** float(formula["h_exp"])  # H in cm
        return float(value)

    def formula_text(self, species: str, use_dem: bool) -> str:
        """Human readable formula, e.g. ``V = exp(-4.798) × W^2.272`` (W, H in cm).

        Args:
            species: Species name or alias.
            use_dem: Which formula to describe.

        Returns:
            The formula text ("" if the species has none).
        """
        formula = self._formula(species, use_dem)
        if formula is None:
            return ""
        coef = f"exp({formula['ln_coef']:g})" if "ln_coef" in formula else f"{formula['coef']:g}"
        parts = [coef]
        if "h_exp" in formula:
            parts.append(f"H^{formula['h_exp']:g}")
        parts.append(f"W^{formula['w_exp']:g}")
        return "V = " + " × ".join(parts)
