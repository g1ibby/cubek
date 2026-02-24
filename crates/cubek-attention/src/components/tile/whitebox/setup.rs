use cubecl::ir::DeviceProperties;
use cubek_matmul::components::CubeDimResource;

use crate::components::tile::TileAttentionFamily;
use crate::components::tile::whitebox::attention::WhiteboxAcceleratedTileAttention;
use crate::components::tile::{SharedTileAttentionConfig, TileAttentionConfig};
use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError, AttentionTileSize,
    InvalidConfigError,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct WhiteboxAcceleratedAttentionMatmulConfig {
    pub shared: SharedTileAttentionConfig,
}

impl TileAttentionConfig for WhiteboxAcceleratedAttentionMatmulConfig {
    fn plane_dim(&self) -> u32 {
        self.shared.plane_dim
    }

    fn num_planes(&self) -> u32 {
        self.shared.num_planes
    }

    fn attention_tile_size(&self) -> AttentionTileSize {
        self.shared.attention_tile_size
    }

    fn num_rows_per_unit(&self) -> u32 {
        self.shared.attention_tile_size.seq_q
    }

    fn causal_mask(&self) -> bool {
        self.shared.causal_mask
    }

    fn materialized_mask(&self) -> bool {
        self.shared.materialized_mask
    }
}

impl TileAttentionFamily for WhiteboxAcceleratedTileAttention {
    type TileAttention<F: AttentionPrecision> = WhiteboxAcceleratedTileAttention;

    type Config = WhiteboxAcceleratedAttentionMatmulConfig;

    fn computation_resources() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Planes(1))
    }

    fn expand_config(
        _device_props: &DeviceProperties,
        blueprint: &AttentionBlueprint,
        _dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        Ok(WhiteboxAcceleratedAttentionMatmulConfig {
            shared: SharedTileAttentionConfig {
                plane_dim: blueprint.plane_dim,
                attention_tile_size: blueprint.tiling_scheme.tile_size,
                num_planes: blueprint.tiling_scheme.stage_size.seq_q,
                causal_mask: blueprint.causal,
                materialized_mask: blueprint.masked,
            },
        })
    }
}
