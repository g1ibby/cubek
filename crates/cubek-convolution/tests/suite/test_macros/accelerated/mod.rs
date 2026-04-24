mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_convolution_accelerated {
    () => {
        mod conv2d_accelerated {
            use super::*;

            #[cfg(all(feature = "conv_tests_plane", not(feature = "conv_tests_mma")))]
            $crate::testgen_convolution_accelerated_algorithm!();

            #[cfg(all(feature = "conv_tests_plane", feature = "conv_tests_mma"))]
            mod cmma {
                use super::*;
                $crate::testgen_convolution_accelerated_algorithm!();
            }

            #[cfg(all(feature = "conv_tests_plane", feature = "conv_tests_mma"))]
            mod mma {
                use super::*;
                $crate::testgen_convolution_accelerated_algorithm!();
            }
        }
    };
}
