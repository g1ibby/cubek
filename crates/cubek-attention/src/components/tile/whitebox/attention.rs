use cubecl;
use cubecl::cmma::MmaDefinition;
use cubecl::prelude::*;
use cubek_matmul::components::tile::StridedTile;

use crate::components::tile::TileAttention;
use crate::components::tile::whitebox::fragment::WhiteboxFragment;
use crate::components::tile::whitebox::fragment::WhiteboxFragmentLayout;
use crate::components::tile::whitebox::setup::WhiteboxAcceleratedAttentionMatmulConfig;
use crate::definition::AttentionPrecision;
use crate::definition::attention_types::*;

/// Uses accelerated instruction, and performs row-dependent computations
/// directly on the fragments
pub struct WhiteboxAcceleratedTileAttention;

#[cube]
impl<AP: AttentionPrecision> TileAttention<AP> for WhiteboxAcceleratedTileAttention {
    type Config = WhiteboxAcceleratedAttentionMatmulConfig;

    type Query = WhiteboxFragment<QT<AP>>;
    type KeyValue = WhiteboxFragment<KVT<AP>>;
    // TODO not sure for mask
    type Mask = WhiteboxFragment<MSK<AP>>;
    type Softmax = WhiteboxFragment<SM<AP>>;
    type SoftmaxRow = WhiteboxFragment<SM<AP>>;
    type Accumulator = WhiteboxFragment<ACC<AP>>;

    type FragmentLayout = WhiteboxFragmentLayout;

    fn softmax_layout(#[comptime] config: Self::Config) -> Self::FragmentLayout {
        let tile_size = config
            .shared
            .attention_tile_size
            .to_score_matmul_tile_size();
        WhiteboxFragmentLayout::new(
            &MmaDefinition::<QT<AP>, KVT<AP>, SM<AP>>::new(
                tile_size.m as usize,
                tile_size.n as usize,
                tile_size.k as usize,
            ),
            cmma::MatrixIdent::Accumulator,
            tile_size,
        )
    }

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::Softmax,
        #[comptime] _config: Self::Config,
    ) {
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {}

    fn allocate_key_value(#[comptime] _config: Self::Config) -> Self::KeyValue {}

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {}

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {}

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {}

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {}

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {}

    fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {}

    fn load_key_transposed<E: Float>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
    }

    fn load_value<E: Float>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
    }

    fn load_mask<E: Numeric>(
        tile: &StridedTile<E>,
        mask: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
    }

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
    }
}
