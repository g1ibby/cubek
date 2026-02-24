use cubecl::cmma::MmaDefinition;
use cubecl::ir::{MatrixIdent, MatrixLayout};
use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords2d;
use cubek_matmul::definition::TileSize;

use crate::components::tile::{
    FragmentAccumulator, FragmentAccumulatorExpand, FragmentLayout, FragmentLayoutExpand,
    FragmentMask, FragmentMaskExpand, FragmentSoftmax, FragmentSoftmaxExpand, RowVal, RowWise,
    RowwiseFormat, RowwiseFormatExpand,
};

/// Returns absolute position in the fragment given a lane and relative row/col within that lane
#[cube]
pub fn absolute_pos_in_fragment(
    lane_id: u32,
    relative_row: u32,
    relative_col: u32,
    #[comptime] mma_line_layout: MatrixLayout,
    #[comptime] mma_line_size: u32,
    #[comptime] ident: MatrixIdent,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) -> (u32, u32) {
    // Assumes elements of a lane are all contiguous on one row or col
    // Which seems to be assumed in MmaDefinition
    // This implies:
    // - When row major, assumes relative_row == 0
    // - When col major, assumes relative_col == 0

    match mma_line_layout {
        MatrixLayout::RowMajor => {
            let contiguous_dim = match ident {
                MatrixIdent::A => k,
                MatrixIdent::B => n,
                MatrixIdent::Accumulator => n,
            };
            let lanes_per_contiguous_dim = contiguous_dim / mma_line_size;
            let row_offset = lane_id / lanes_per_contiguous_dim;
            let col_offset = (lane_id % lanes_per_contiguous_dim) * mma_line_size;

            // Assuming row_offset = 0
            (row_offset, col_offset + relative_col)
        }
        MatrixLayout::ColMajor => {
            let contiguous_dim = match ident {
                MatrixIdent::A => m,
                MatrixIdent::B => k,
                MatrixIdent::Accumulator => m,
            };
            let lanes_per_contiguous_dim = contiguous_dim / mma_line_size;
            let row_offset = (lane_id % lanes_per_contiguous_dim) * mma_line_size;
            let col_offset = lane_id / lanes_per_contiguous_dim;

            // Assuming col_offset = 0
            (row_offset + relative_row, col_offset)
        }
        MatrixLayout::Undefined => unimplemented!("Unsupported layout"),
    }
}

// Assumes elements of a lane are all contiguous on one row or col
// Which seems to be assumed in MmaDefinition
pub fn num_units_per_row(
    mma_line_layout: MatrixLayout,
    mma_line_size: u32,
    n: u32,
    k: u32,
    ident: MatrixIdent,
) -> u32 {
    match (ident, mma_line_layout) {
        (MatrixIdent::A, MatrixLayout::RowMajor) => k / mma_line_size,
        (MatrixIdent::A, MatrixLayout::ColMajor) => k,
        (MatrixIdent::B, MatrixLayout::RowMajor) => n / mma_line_size,
        (MatrixIdent::B, MatrixLayout::ColMajor) => n,
        (MatrixIdent::Accumulator, MatrixLayout::RowMajor) => n / mma_line_size,
        (MatrixIdent::Accumulator, MatrixLayout::ColMajor) => n,
        (_, _) => unimplemented!("Unsupported layout"),
    }
}

#[derive(CubeType)]
pub struct WhiteboxFragmentLayout {
    #[cube(comptime)]
    mma_line_layout: MatrixLayout,
    #[cube(comptime)]
    mma_line_size: u32,
    #[cube(comptime)]
    mma_lines_per_lane: u32,
    #[cube(comptime)]
    matrix_ident: MatrixIdent,
    #[cube(comptime)]
    tile_size: TileSize,
}

#[cube]
impl WhiteboxFragmentLayout {
    pub fn new<A: CubePrimitive, B: CubePrimitive, CD: CubePrimitive>(
        mma_def: &MmaDefinition<A, B, CD>,
        #[comptime] matrix_ident: MatrixIdent,
        #[comptime] tile_size: TileSize,
    ) -> Self {
        let mma_line_size = mma_def.line_size(matrix_ident);
        let mma_lines_per_lane = mma_def.lines_per_lane(matrix_ident);
        WhiteboxFragmentLayout {
            mma_line_layout: mma_def.line_layout(matrix_ident),
            mma_line_size: comptime!(mma_line_size as u32),
            mma_lines_per_lane: comptime!(mma_lines_per_lane as u32),
            matrix_ident,
            tile_size,
        }
    }

    pub fn absolute_index(&self, local_pos: Coords2d) -> u32 {
        match comptime!(self.mma_line_layout) {
            MatrixLayout::RowMajor => local_pos.1,
            MatrixLayout::ColMajor => local_pos.0,
            MatrixLayout::Undefined => unimplemented!(),
        }
    }

    pub fn num_local_rows(&self) -> comptime_type!(u32) {
        comptime! {
            match self.mma_line_layout {
                MatrixLayout::RowMajor => 1,
                MatrixLayout::ColMajor => self.mma_lines_per_lane * self.mma_line_size,
                MatrixLayout::Undefined => unimplemented!(),
            }
        }
    }

    pub fn num_local_cols(&self) -> comptime_type!(u32) {
        comptime! {
        match self.mma_line_layout {
            MatrixLayout::RowMajor => self.mma_lines_per_lane * self.mma_line_size,
            MatrixLayout::ColMajor => 1,
            MatrixLayout::Undefined => unimplemented!(),
        }
            }
    }
}

#[cube]
impl FragmentLayout for WhiteboxFragmentLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        absolute_pos_in_fragment(
            UNIT_POS_PLANE,
            local_pos.0,
            local_pos.1,
            self.mma_line_layout,
            self.mma_line_size,
            self.matrix_ident,
            self.tile_size.m,
            self.tile_size.n,
            self.tile_size.k,
        )
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        comptime!(num_units_per_row(
            self.mma_line_layout,
            self.mma_line_size,
            self.tile_size.n,
            self.tile_size.k,
            self.matrix_ident,
        ))
    }
}

#[derive(CubeType)]
pub struct WhiteboxFragment<E: Numeric> {
    array: Array<Line<E>>,
    layout: WhiteboxFragmentLayout,
}

#[cube]
impl<E: Float> WhiteboxFragment<E> {
    fn zero(&mut self) {
        for i in 0..self.layout.mma_lines_per_lane {
            self.array[i as usize] = Line::cast_from(0)
        }
    }
}

#[cube]
impl<E: Float> FragmentSoftmax<E> for WhiteboxFragment<E> {
    type Layout = WhiteboxFragmentLayout;

    type SoftmaxRowFormat = WhiteboxFragment<E>;

    fn rowwise_mut(&mut self) -> &mut WhiteboxFragment<E> {
        self
    }

    fn update_from_rowwise(&mut self) {
        // Nothing to do, because rowwise = self
    }

    fn zero(&mut self) {
        self.zero()
    }
}

#[cube]
impl<E: Float> RowwiseFormat<E> for WhiteboxFragment<E> {
    type Layout = WhiteboxFragmentLayout;

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        self.layout.num_units_per_row()
    }

    fn rowwise_max(&self) -> RowWise<E> {
        match comptime!(self.layout.mma_line_layout) {
            // We take the max of all values
            MatrixLayout::RowMajor => {
                let mut val = E::min_value();
                #[unroll]
                for l in 0..self.layout.mma_lines_per_lane {
                    let ln = self.array[l as usize];
                    for e in 0..self.layout.mma_line_size {
                        val = max(ln[e as usize], val);
                    }
                }

                let mut vals = Sequence::new();
                vals.push(RowVal::<E> { val });
                RowWise::<E> {
                    num_rows: self.layout.num_local_rows() as usize,
                    vals,
                }
            }
            // All values are their own max
            MatrixLayout::ColMajor => {
                let mut vals = Sequence::new();
                #[unroll]
                for l in 0..self.layout.mma_lines_per_lane {
                    #[unroll]
                    for e in 0..self.layout.mma_line_size {
                        vals.push(RowVal::<E> {
                            val: self.array[l as usize][e as usize],
                        });
                    }
                }

                RowWise::<E> {
                    num_rows: self.layout.num_local_rows() as usize,
                    vals,
                }
            }
            MatrixLayout::Undefined => unimplemented!(),
        }
    }

    fn rowwise_sum(&self) -> RowWise<E> {
        todo!()
    }

    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M) {
        todo!()
    }

    fn exp_diff(&mut self, m: &RowWise<E>) {
        todo!()
    }
}

#[cube]
impl<E: Float> FragmentAccumulator<E> for WhiteboxFragment<E> {
    fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        match comptime!(self.layout.mma_line_layout) {
            // All lines are directly scales by the same scale
            MatrixLayout::RowMajor => {
                let row_scale = scale.index(0);
                #[unroll]
                for l in 0..self.layout.mma_lines_per_lane {
                    self.array[l as usize] = self.array[l as usize] * Line::cast_from(row_scale);
                }
            }
            // All elements are on a different row
            MatrixLayout::ColMajor =>
            {
                #[unroll]
                for l in 0..self.layout.mma_lines_per_lane {
                    let row_offset = l * self.layout.mma_line_size;
                    #[unroll]
                    for e in 0..self.layout.mma_line_size {
                        let row_scale = scale.index((row_offset + e) as usize);
                        self.array[l as usize][e as usize] =
                            self.array[l as usize][e as usize] * row_scale;
                    }
                }
            }
            MatrixLayout::Undefined => unimplemented!(),
        }
    }

    fn zero(&mut self) {
        self.zero()
    }
}

#[cube]
impl<E: Numeric> FragmentMask for WhiteboxFragment<E> {
    type Layout = WhiteboxFragmentLayout;

    fn should_mask(&self, local_pos: Coords2d) -> bool {
        bool::cast_from(self.array[self.layout.absolute_index(local_pos) as usize])
    }
}
