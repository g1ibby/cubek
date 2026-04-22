use crate::components::tile_matmul::TileMatmulFamily;
use crate::components::tile_matmul::dispatch::DispatchTileMatmul;
use crate::components::tile_matmul::dispatch::config::DispatchConfig;
use crate::{components::resource::CubeDimResource, components::tile_matmul::Plane};
use crate::{
    definition::{MatmulAvailabilityError, MatmulSetupError, MatmulVectorSizes},
    definition::{MatmulElems, TilingBlueprint},
};
use cubecl::{
    {features::MmaConfig, ir::DeviceProperties},
    {ir::StorageType, prelude::*},
};
use cubek_std::{InvalidConfigError, TileSize};

impl TileMatmulFamily for DispatchTileMatmul {
    type Config = DispatchConfig;
    type Scope = Plane;
    type Matmul<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size> =
        DispatchTileMatmul;

    fn requires_accelerator() -> bool {
        todo!()
        // match self;
    }

    fn can_cast_stage_element() -> bool {
        false
    }

    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Planes(1))
    }

    fn expand_config(
        _device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<DispatchConfig, MatmulSetupError> {
        todo!()
    }

    fn should_swizzle<R: Runtime>(_client: &ComputeClient<R>) -> bool {
        // Unsupported
        false
    }

    fn is_supported<R: Runtime>(client: &ComputeClient<R>, config: MmaConfig) -> bool {
        client.properties().features.matmul.cmma.contains(&config)
    }

    fn supported_sizes<R: Runtime>(
        client: &ComputeClient<R>,
        lhs_ty: StorageType,
        rhs_ty: StorageType,
        acc_ty: StorageType,
    ) -> Vec<TileSize> {
        client
            .properties()
            .features
            .matmul
            .cmma
            .iter()
            .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
            .map(|it| (it.m, it.n, it.k).into())
            .collect()
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        let lhs = dtypes.lhs_register;
        let rhs = dtypes.rhs_register;
        let acc = dtypes.acc_register;

        let size = blueprint.tiling_scheme.tile_size;
        if !client
            .properties()
            .features
            .matmul
            .cmma
            .contains(&MmaConfig {
                a_type: lhs,
                b_type: rhs,
                cd_type: acc,
                m: size.m(),
                k: size.k(),
                n: size.n(),
            })
        {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs,
                    rhs,
                    output: acc,
                    size: Some(TileSize::new(size.m(), size.n(), size.k())),
                },
            ));
        }

        if blueprint.swizzle_modes.has_swizzle() {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "This tile matmul doesn't support swizzling",
            )));
        }

        Ok(())
    }
}
