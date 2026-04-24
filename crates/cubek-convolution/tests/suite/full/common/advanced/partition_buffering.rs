// Partition-buffering sweep (Single + Double) is intentionally NOT done here
// to keep the full-tier cartesian within budget — the previous feature-gated
// behavior excluded `conv_tests_partition_buffering` from `full`. The Double
// variant is exercised once on a representative routine in
// `extended/advanced.rs`.

#[macro_export]
macro_rules! testgen_convolution_partition_buffering {
    ($algorithm: ty, $dtypes: expr, $tiling_scheme: expr, $swizzle: expr) => {
        use cubek_matmul::components::stage::PartitionBuffering;

        $crate::testgen_convolution_problem!(
            $algorithm,
            $dtypes,
            $tiling_scheme,
            $swizzle,
            PartitionBuffering::Single
        );
    };
}
