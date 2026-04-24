use cubecl::frontend::CubePrimitive;
use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime, cube, ir::StorageType, prelude::*,
    std::tensor::TensorHandle, zspace::Shape,
};
use cubek_reduce::components::instructions::plane_topk_merge;
use cubek_test_utils::{DataKind, InputDataType, StrideSpec, TestInput};

use crate::it::reference::contiguous_strides;
use cubecl::frontend::CompilationArg;

#[test]
fn test_plane_reduce_inplace() {
    let client = TestRuntime::client(&Default::default());

    // plane_size of 16 with vector_size of 4
    let num_threads = 2;
    let k = 2;
    let vector_size = 4;
    let total_vectors = num_threads * k * vector_size;

    let shape = Shape::new([total_vectors]);
    let stride = contiguous_strides(&shape);

    let dtype = f32::as_type_native_unchecked().storage_type();
    let input_dtype = InputDataType::Standard(dtype);

    #[rustfmt::skip]
    let data = vec![
        // Thread 0
        99.0, 99.1, 99.2, 99.3, 
        10.0, 10.1, 10.2, 10.3, 
        // Thread 1
        88.0, 88.1, 102.2, 88.3, 
        55.0, 55.1, 101.2, 55.3,
    ];

    let (input_handle, _input_host) = TestInput::new(
        client.clone(),
        shape.clone(),
        input_dtype,
        StrideSpec::Custom(stride.iter().copied().collect()),
        DataKind::Custom { data: data.clone() },
    )
    .generate_with_f32_host_data();

    let storage_type = f32::as_type_native_unchecked().storage_type();

    let output_handle = build_output_tensor(&client, storage_type, &shape);

    launch_plane_reduce_inplace::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, num_threads),
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
        k,
        storage_type,
        vector_size,
    );

    let bytes = client.read_one(output_handle.handle).unwrap();
    let actual = f32::from_bytes(&bytes);
    assert_plane_topk_custom_values(&data, actual, num_threads, k, vector_size);
}

fn build_output_tensor(
    client: &cubecl::client::ComputeClient<TestRuntime>,
    output_dtype: StorageType,
    output_shape: &Shape,
) -> TensorHandle<TestRuntime> {
    let strides = contiguous_strides(output_shape);
    TestInput::new(
        client.clone(),
        output_shape.clone(),
        output_dtype,
        StrideSpec::Custom(strides.iter().copied().collect()),
        DataKind::Zeros,
    )
    .generate()
}

#[cube(launch)]
fn launch_plane_reduce_inplace<N: Numeric, S: Size>(
    input: &Tensor<Vector<N, S>>,
    output: &mut Tensor<Vector<N, S>>,
    #[comptime] k: usize,
    #[define(N)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    let mut elements = Array::new(k);
    let offset = UNIT_POS_X as usize * k;

    // Load backwards so the accumulator is already sorted descending locally
    #[unroll]
    for i in 0..k {
        elements[i] = input[offset + i];
    }

    plane_topk_merge::<N, S>(k, &mut elements);

    #[unroll]
    for i in 0..k {
        output[offset + i] = elements[i];
    }
}

fn assert_plane_topk_custom_values(
    input_host: &[f32],
    actual_gpu: &[f32],
    num_threads: usize,
    k: usize,
    vector_size: usize,
) {
    let mut expected_topk = vec![0.0; k * vector_size];

    // Sort each lane independently
    for lane in 0..vector_size {
        let mut lane_values = Vec::new();
        for i in 0..(num_threads * k) {
            lane_values.push(input_host[i * vector_size + lane]);
        }

        // Sort descending for this specific lane
        lane_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        for i in 0..k {
            expected_topk[i * vector_size + lane] = lane_values[i];
        }
    }

    for unit in 0..num_threads {
        let start = unit * k * vector_size;
        let end = start + (k * vector_size);
        assert_eq!(&actual_gpu[start..end], expected_topk.as_slice());
    }
}
