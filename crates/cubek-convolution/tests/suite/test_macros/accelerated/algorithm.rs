#[macro_export]
macro_rules! testgen_convolution_accelerated_algorithm {
    () => {
        use cubek_convolution::components::global::read::strategy::{
            async_full_cyclic, async_full_strided,
        };
        use cubek_convolution::{
            kernels::algorithm::simple::*, kernels::algorithm::specialized::*,
        };
        use cubek_matmul::components::global::read::{
            sync_full_cyclic, sync_full_strided, sync_full_tilewise,
        };
        use cubek_matmul::components::stage::{ColMajorTilingOrder, RowMajorTilingOrder};

        #[cfg(all(feature = "conv_tests_simple", feature = "conv_tests_cyclic"))]
        mod simple_cyclic {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleSyncCyclicConv);
        }

        #[cfg(all(feature = "conv_tests_simple", feature = "conv_tests_strided"))]
        mod simple_strided {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleSyncStridedConv);
        }

        #[cfg(all(feature = "conv_tests_simple", feature = "conv_tests_tilewise"))]
        mod simple_tilewise {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleSyncTilewiseConv);
        }

        #[cfg(all(
            feature = "conv_tests_simple",
            feature = "conv_tests_cyclic",
            feature = "conv_tests_async_copy"
        ))]
        mod simple_async_cyclic {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleAsyncCyclicConv);
        }

        #[cfg(all(
            feature = "conv_tests_simple",
            feature = "conv_tests_strided",
            feature = "conv_tests_async_copy"
        ))]
        mod simple_async_strided {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleAsyncStridedConv);
        }

        #[cfg(all(feature = "conv_tests_simple", feature = "conv_tests_tma"))]
        mod simple_tma {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SimpleAsyncTmaConv);
        }

        // Broken and can't currently figure out why

        // #[cfg(all(
        //     feature = "conv_tests_specialized",
        //     feature = "conv_tests_cyclic",
        //     feature = "conv_tests_async_copy"
        // ))]
        // mod specialized_async_cyclic {
        //     use super::*;

        //     $crate::testgen_convolution_accelerated_precision!(SpecializedCyclicConv<TMM>);
        // }

        // #[cfg(all(
        //     feature = "conv_tests_specialized",
        //     feature = "conv_tests_strided",
        //     feature = "conv_tests_async_copy"
        // ))]
        // mod specialized_async_strided {
        //     use super::*;

        //     $crate::testgen_convolution_accelerated_precision!(SpecializedStridedConv<TMM>);
        // }

        #[cfg(all(feature = "conv_tests_specialized", feature = "conv_tests_tma"))]
        mod specialized_tma {
            use super::*;

            $crate::testgen_convolution_accelerated_precision!(SpecializedTmaConv);
        }
    };
}
