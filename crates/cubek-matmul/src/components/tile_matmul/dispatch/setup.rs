use crate::components::tile_matmul::dispatch::DispatchTileMatmul;
use crate::components::tile_matmul::dispatch::config::DispatchConfig;
use crate::components::tile_matmul::mma::config::MmaMatmulConfig;
use crate::components::tile_matmul::{InterleavedMatmulConfig, TileMatmulFamily};
use crate::components::tile_matmul::{
    PlaneVecMatInnerProductConfig, RegisterMatmulConfig, SharedTileConfig,
};
use crate::{components::resource::CubeDimResource, components::tile_matmul::Plane};
use crate::{
    definition::{MatmulAvailabilityError, MatmulSetupError, MatmulVectorSizes},
    definition::{MatmulElems, TilingBlueprint},
};
use cubecl::{
    {features::MmaConfig, ir::DeviceProperties},
    {ir::StorageType, prelude::*},
};
use cubek_std::tile::mma::MmaIOConfig;
use cubek_std::{InvalidConfigError, TileSize};

impl TileMatmulFamily for DispatchTileMatmul {
    type Config = DispatchConfig;
    type Scope = Plane;
    type Matmul<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size> =
        DispatchTileMatmul;

    fn requires_accelerator(&self) -> bool {
        match self {
            DispatchTileMatmul::Cmma | DispatchTileMatmul::Mma => true,
            DispatchTileMatmul::Register
            | DispatchTileMatmul::PlaneVec
            | DispatchTileMatmul::Interleaved => false,
        }
    }

    fn can_cast_stage_element(&self) -> bool {
        match self {
            DispatchTileMatmul::Cmma => false,
            DispatchTileMatmul::Mma
            | DispatchTileMatmul::Register
            | DispatchTileMatmul::PlaneVec
            | DispatchTileMatmul::Interleaved => true,
        }
    }

    fn should_swizzle<R: Runtime>(&self, client: &ComputeClient<R>) -> bool {
        match self {
            DispatchTileMatmul::Cmma => {
                // Unsupported
                false
            }
            DispatchTileMatmul::Mma => {
                // No alignment means swizzling can't be properly used, since it needs to be applied to
                // the address, and alignment guarantees the offset is aligned to the pattern repeat.
                client.properties().features.alignment
            }
            DispatchTileMatmul::Register => {
                // Selection isn't getting rid of all conflicts with the current load strategy, but does
                // reduce conflicts significantly (i.e. average 18 vs average 5). Should try to find more
                // optimal settings in the future.
                client.properties().features.alignment
            }
            DispatchTileMatmul::PlaneVec => {
                // Supported but need to find good settings for this tiling. Currently tuned for `ldmatrix`.
                // Need to profile at some point
                false
            }
            DispatchTileMatmul::Interleaved => {
                // Selection isn't getting rid of all conflicts with the current load strategy, but does
                // reduce conflicts significantly (i.e. average 18 vs average 5). Should try to find more
                // optimal settings in the future.
                client.properties().features.alignment
            }
        }
    }

    fn cubedim_resource(&self) -> Result<CubeDimResource, InvalidConfigError> {
        match self {
            DispatchTileMatmul::Cmma
            | DispatchTileMatmul::Mma
            | DispatchTileMatmul::PlaneVec
            | DispatchTileMatmul::Interleaved => Ok(CubeDimResource::Planes(1)),
            DispatchTileMatmul::Register => Ok(CubeDimResource::Units(1)),
        }
    }

    fn expand_config(
        &self,
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(match self {
            DispatchTileMatmul::Cmma => DispatchConfig::Cmma(SharedTileConfig::new(
                blueprint.tiling_scheme.tile_size,
                blueprint.plane_dim,
                blueprint.swizzle_modes,
            )),
            DispatchTileMatmul::Mma => DispatchConfig::Mma(MmaMatmulConfig {
                shared: SharedTileConfig {
                    tile_size: blueprint.tiling_scheme.tile_size,
                    plane_dim: blueprint.plane_dim,
                    swizzle_modes: blueprint.swizzle_modes,
                },
                mma_io_config: MmaIOConfig::new(
                    device_props,
                    dtypes.lhs_stage,
                    dtypes.rhs_stage,
                    dtypes.acc_stage,
                ),
            }),
            DispatchTileMatmul::Register => {
                DispatchConfig::Register(RegisterMatmulConfig::from_shared_tile_config(
                    blueprint.lhs_layout,
                    blueprint.rhs_layout,
                    SharedTileConfig::new(
                        blueprint.tiling_scheme.tile_size,
                        blueprint.plane_dim,
                        blueprint.swizzle_modes,
                    ),
                ))
            }
            DispatchTileMatmul::PlaneVec => {
                DispatchConfig::PlaneVec(PlaneVecMatInnerProductConfig::new(
                    SharedTileConfig::new(
                        blueprint.tiling_scheme.tile_size,
                        blueprint.plane_dim,
                        blueprint.swizzle_modes,
                    ),
                    vector_sizes.lhs as u32,
                ))
            }
            DispatchTileMatmul::Interleaved => DispatchConfig::Interleaved(
                InterleavedMatmulConfig::from_shared_tile_config(SharedTileConfig::new(
                    blueprint.tiling_scheme.tile_size,
                    blueprint.plane_dim,
                    blueprint.swizzle_modes,
                )),
            ),
        })
    }

    fn is_supported<R: Runtime>(&self, client: &ComputeClient<R>, config: MmaConfig) -> bool {
        match self {
            DispatchTileMatmul::Cmma => client.properties().features.matmul.cmma.contains(&config),
            DispatchTileMatmul::Mma => client.properties().features.matmul.mma.contains(&config),
            DispatchTileMatmul::Register
            | DispatchTileMatmul::PlaneVec
            | DispatchTileMatmul::Interleaved => true,
        }
    }

    fn supported_sizes<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        lhs_ty: StorageType,
        rhs_ty: StorageType,
        acc_ty: StorageType,
    ) -> Vec<TileSize> {
        let iters = match self {
            DispatchTileMatmul::Cmma => &client.properties().features.matmul.cmma,
            DispatchTileMatmul::Mma => &client.properties().features.matmul.mma,
            DispatchTileMatmul::Register
            | DispatchTileMatmul::PlaneVec
            | DispatchTileMatmul::Interleaved => return Vec::new(),
        };

        iters
            .iter()
            .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
            .map(|it| (it.m, it.n, it.k).into())
            .collect()
    }

    fn validate_blueprint<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
    }
}
