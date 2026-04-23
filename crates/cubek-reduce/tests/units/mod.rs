use cubecl::frontend::CubePrimitive;
use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime, cube,
    ir::StorageType,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_reduce::ReducePrecision;
use cubek_reduce::components::instructions::{Accumulator, Value, plane_reduce_inplace};
use cubek_test_utils::{DataKind, InputDataType, StrideSpec, TestInput};

use crate::it::reference::contiguous_strides;
use cubecl::frontend::CompilationArg;

#[test]
fn test_plane_reduce_inplace() {
    let client = TestRuntime::client(&Default::default());
    let k = 2;
    let num_threads = 2;

    // We need (num_threads * k) vectors total
    let total_elements = num_threads * k;
    let shape = Shape::new([total_elements]);

    let stride = contiguous_strides(&shape);

    let input_dtype = InputDataType::Standard(f32::as_type_native_unchecked().storage_type());

    let (input_handle, _input_host) = TestInput::new(
        client.clone(),
        shape.clone(),
        input_dtype,
        StrideSpec::Custom(stride.iter().copied().collect()),
        DataKind::Arange { scale: Some(1.) },
    )
    .generate_with_f32_host_data();

    let storage_type = f32::as_type_native_unchecked().storage_type();

    let output_handle = build_output_tensor(&client, storage_type, &shape);

    launch_plane_reduce_inplace::launch::<f32, TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, num_threads),
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
        k
    );

    let bytes = client.read_one(output_handle.handle).unwrap();
    let actual = f32::from_bytes(&bytes);

    let expected = vec![3.0, 2.0, 3.0, 2.0];
    assert_eq!(actual, expected, "The plane reduction did not converge to the global Top-K");
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
fn launch_plane_reduce_inplace<P: ReducePrecision + Send + Sync>(
    input: &Tensor<Vector<P::EI, P::SI>>,
    output: &mut Tensor<Vector<P::EA, P::SI>>,
    #[comptime] k: usize,
) {
    let mut elements = Array::new(k);

    let offset = UNIT_POS_X as usize * k;

    #[unroll]
    for i in 0..k {
        // Use cast_from but ensure it knows it's targeting Vector<P::EA, P::SI>
        elements[i] = Vector::<P::EA, P::SI>::cast_from(input[offset + i]);
    }
    let mut accumulator = Accumulator::<P> {
        elements: Value::new_Multiple(elements),
        args: Value::new_None(),
    };

    plane_reduce_inplace::<P>(k, &mut accumulator);

    let final_elements = accumulator.elements.multiple();
    #[unroll]
    for i in 0..k {
        output[offset + i] = final_elements[i];
    }
}
