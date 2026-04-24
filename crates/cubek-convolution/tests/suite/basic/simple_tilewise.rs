//! Smoke tests for `SimpleSyncTilewiseConv`.

use cubek_convolution::kernels::algorithm::simple::SimpleSyncTilewiseConv;
use cubek_matmul::components::tile_matmul::cmma::CmmaMatmul;

use super::common::{
    default_partition_buffering, default_swizzle, default_tiling_scheme, f16_dtypes, small_size,
};
use crate::suite::launcher_strategy::test_algo;

#[test]
fn simple_tilewise_cmma_small_f16() {
    test_algo::<SimpleSyncTilewiseConv<CmmaMatmul>>(
        f16_dtypes(),
        default_tiling_scheme(),
        default_swizzle(),
        default_partition_buffering(),
        small_size(),
    );
}
