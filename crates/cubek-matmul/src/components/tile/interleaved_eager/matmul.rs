use cubecl::prelude::*;

use crate::components::tile::TileConfig as _;
use crate::components::tile::interleaved_eager::config::InterleavedEagerMatmulConfig;
use crate::components::tile::interleaved_eager::reader::InterleavedStageReader;
use crate::components::tile::interleaved_eager::writer::InterleavedStageWriter;
use crate::components::tile::io::Strided;
use crate::components::tile::tile_data::StridedTile;
use crate::components::tile::{TileMatmul, io::Filled};
use crate::definition::{MatrixLayout, StageIdent};

/// Computes a tile matmul where each unit of the plane accumulates an interleaved (by plane_dim)
/// partial dot-product over K.
///
/// The plane combines those contributions at every element.
/// The caveat is that each a plane_sum is called for every accumulator element.
pub struct InterleavedEagerMatmul {}

#[derive(CubeType)]
/// InterleavedFragment: each unit owns a stripe of the input tile.
pub struct InterleavedEagerFragment<E: Numeric> {
    pub array: Array<E>,
    #[cube(comptime)]
    pub layout: MatrixLayout,
    #[cube(comptime)]
    row_count: usize,
    #[cube(comptime)]
    col_count: usize,
}

#[cube]
impl<E: Numeric> InterleavedEagerFragment<E> {
    fn get(&self, i: usize, j: usize) -> E {
        match comptime!(self.layout) {
            MatrixLayout::RowMajor => self.array[i * self.col_count + j],
            MatrixLayout::ColMajor => self.array[j * self.row_count + i],
        }
    }
}

#[derive(CubeType)]
/// InterleavedAccumulator: each unit holds a full accumulator with partial K contributions,
/// combined later via `consolidate`.
pub struct InterleavedEagerAccumulator<E: Numeric> {
    pub array: Array<E>,
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for InterleavedEagerMatmul {
    type Config = InterleavedEagerMatmulConfig;

    // Size m * k_local
    type LhsFragment = InterleavedEagerFragment<L>;
    // Size k_local * n
    type RhsFragment = InterleavedEagerFragment<R>;
    // Size m * n
    type AccFragment = InterleavedEagerAccumulator<A>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Filled;
    type OutTile = Strided;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        let m = config.elements_per_unit_m();
        let n = config.elements_per_unit_n();
        let local_k = config.elements_per_unit_k();

        #[unroll]
        for m_ in 0..m {
            #[unroll]
            for n_ in 0..n {
                let mut current_acc = A::cast_from(0);

                #[unroll]
                for k_ in 0..local_k {
                    let lhs_elem = A::cast_from(lhs.get(m_, k_));
                    let rhs_elem = A::cast_from(rhs.get(k_, n_));
                    current_acc += lhs_elem * rhs_elem;
                }

                let plane_summed = plane_sum(current_acc);

                let acc_index = m_ * n + n_;
                let plane_dim = config.plane_dim() as usize;
                let local_acc_index = acc_index / plane_dim;
                let storing_unit = acc_index % plane_dim;

                let is_storing_unit = A::cast_from(UNIT_POS_X as usize == storing_unit);
                acc.array[local_acc_index] += is_storing_unit * plane_summed;
            }
        }
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        let row_count = config.elements_per_unit_m();
        let col_count = config.elements_per_unit_k();
        InterleavedEagerFragment::<L> {
            array: Array::new(row_count * col_count),
            layout,
            row_count,
            col_count,
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        let row_count = config.elements_per_unit_k();
        let col_count = config.elements_per_unit_n();
        InterleavedEagerFragment::<R> {
            array: Array::new(row_count * col_count),
            layout,
            row_count,
            col_count,
        }
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        // hardcoded to row major right now
        assert!(matches!(layout, MatrixLayout::RowMajor));
        InterleavedEagerAccumulator::<A> {
            array: Array::new(config.num_local_accumulators()),
        }
    }

    fn load_lhs<E: Numeric>(
        tile: &StridedTile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
        InterleavedStageReader::load_fragment(tile, lhs, StageIdent::Lhs, config);
    }

    fn load_rhs<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        InterleavedStageReader::load_fragment(tile, rhs, StageIdent::Rhs, config);
    }

    fn load_acc<E: Numeric>(
        tile: &E,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        InterleavedStageReader::load_accumulator::<A, E>(tile, acc, config);
    }

    fn write_results<E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        InterleavedStageWriter::store_fragment(tile, acc, config)
    }
}
