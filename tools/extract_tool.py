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

    type_name: str = Field(..., description="苗木类型名称，如 罗汉松、香樟")
    image_id: Optional[str] = Field(None, description="图像id，省略则使用当前图像")


class ExtractTool(BaseTool):
    """Segment all trees of a type and add them to the map."""

    name = "extract_trees"
    description = (
        "在图像中按苗木类型名称调用SAM3分割提取苗木：合并所有窗口结果为栅格，填补空洞、清除碎屑后做连通域分析得到数量，结果保存到会话并作为图层显示在地图上。"
        "统计、估值、制表、报告都依赖该工具的结果。"
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
            raise ToolError("苗木类型名称不能为空。")
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
            return ToolResult(message=f"在「{image.name}」中没有检测到**{species}**。", data={"count": 0})

        layer_name = f"{species} ({image.name})"
        ctx.session.layers = [l for l in ctx.session.layers if l.name != layer_name]
        layer = ctx.services.layer_store.save(gdf, layer_name)
        layer.image_id, layer.species = image.image_id, species
        ctx.session.layers.append(layer)
        ctx.session.touch()
        note = "" if image.georeferenced else "\n\n> ⚠️ 该图像无地理坐标，冠幅/面积按 1 像素 = 配置的像元尺寸（默认 1 米）估算，仅供参考。"
        return ToolResult(
            message=f"已在「{image.name}」中提取 **{len(gdf)}** 株{species}（提示词 `{prompt}`），已添加到地图图层「{layer_name}」。{note}",
            data={
                "count": len(gdf), "species": species, "layer_id": layer.layer_id,
                "raw_instances": result.raw_instances, "raw_components": stats.raw_components,
                "filled_holes": stats.filled_holes, "removed_debris": stats.removed_debris,
            },
        )
