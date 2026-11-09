"""Tool 1: extract nursery stock of a given type with SAM3."""

from __future__ import annotations

import logging
import pandas as pd
import geopandas as gpd
from typing import Optional

from pydantic import BaseModel, Field

from tools._common import resolve_image
from tools.base import BaseTool, ToolContext, ToolError, ToolResult

logger = logging.getLogger(__name__)


class ExtractArgs(BaseModel):
    """Arguments of :class:`ExtractTool`."""

    type_name: str = Field(..., description="Tree type name, e.g. Buddhist Pine, Camphor Tree")
    image_id: Optional[str] = Field(None, description="Image id; omit to use the current image")


class ExtractTool(BaseTool):
    """Segment all trees of a type and add them to the map."""

    name = "extract_trees"
    description = (
        "Extract the trees of a given type from the image with SAM3: the results of all windows are merged into one raster, "
        "holes are filled and debris removed, then connected-component analysis gives the count. The result is stored in the "
        "session and shown on the map as a layer. Counting, valuation, tables and reports all depend on this tool."
    )
    args_schema = ExtractArgs

    def run(self, ctx: ToolContext, args: ExtractArgs) -> ToolResult:
        """Run SAM3 and store the instances.

        Args:
            ctx: Tool context.
            args: Validated arguments.

        Returns:
            Result with the number of extracted trees.

        Raises:
            ToolError: If the image is missing.
        """
        image = resolve_image(ctx, args.image_id)
        pricing = ctx.services.pricing
        species = pricing.resolve(args.type_name) or args.type_name.strip()
        if not species:
            raise ToolError("The tree type name must not be empty.")
        prompt = pricing.prompt_for(args.type_name)
        logger.info("Extract %s (prompt=%r) from %s", species, prompt, image.name)

        result = ctx.services.segmenter.segment(image.path, prompt, image.rescale)
        gdf, stats = result.trees, result.stats
        gdf["species"] = species

        old = ctx.session.trees.get(image.image_id)
        if old is not None and not old.empty:
            old = old[old["species"] != species]
            gdf_all = gpd.GeoDataFrame(pd.concat([old, gdf], ignore_index=True), crs="EPSG:4326")
        else:
            gdf_all = gdf
        ctx.session.trees[image.image_id] = gdf_all

        if gdf.empty:
            return ToolResult(message=f"No **{species}** detected in “{image.name}”.", data={"count": 0})

        layer_name = f"{species} ({image.name})"
        ctx.session.layers = [l for l in ctx.session.layers if l.name != layer_name]
        layer = ctx.services.layer_store.save(gdf, layer_name)
        layer.image_id, layer.species = image.image_id, species
        ctx.session.layers.append(layer)
        ctx.session.touch()
        note = "" if image.georeferenced else "\n\n> ⚠️ This image has no geographic coordinates; crown width / area are estimated with 1 pixel = the configured pixel size (default 1 m) and are for reference only."
        return ToolResult(
            message=f"Extracted **{len(gdf)}** trees of {species} in “{image.name}” (prompt `{prompt}`); added as map layer “{layer_name}”.{note}",
            data={
                "count": len(gdf), "species": species, "layer_id": layer.layer_id,
                "raw_instances": result.raw_instances, "raw_components": stats.raw_components,
                "filled_holes": stats.filled_holes, "removed_debris": stats.removed_debris,
            },
        )
