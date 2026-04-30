use cubecl::{
    CubeElement, Runtime, TestRuntime, frontend::CubePrimitive, std::tensor::TensorHandle,
};
use cubek_fft::{irfft_launch_padded, rfft_launch_padded};
use cubek_test_utils::{HostData, HostDataType, HostDataVec};

fn to_f32(host: HostData) -> Vec<f32> {
    match host.data {
        HostDataVec::F32(v) => v,
        _ => panic!("expected f32 host data"),
    }
}

fn empty_tensor(
    client: &cubecl::client::ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dtype: cubecl::prelude::StorageType,
) -> TensorHandle<TestRuntime> {
    let elems = shape.iter().product::<usize>();
    TensorHandle::<TestRuntime>::new_contiguous(shape, client.empty(elems * dtype.size()), dtype)
}

#[test]
fn rfft_signal_len_matches_materialized_zero_padding() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let n_fft = 8usize;
    let signal_len = 5usize;
    let shape = vec![1, n_fft];
    let n_freq = n_fft / 2 + 1;

    let virtual_data = vec![0.25, -0.5, 1.0, 0.75, -0.125, 9.0, -7.0, 3.0];
    let mut padded_data = virtual_data.clone();
    for value in padded_data.iter_mut().take(n_fft).skip(signal_len) {
        *value = 0.0;
    }

    let virtual_signal = TensorHandle::<TestRuntime>::new_contiguous(
        shape.clone(),
        client.create_from_slice(f32::as_bytes(&virtual_data)),
        dtype,
    );
    let padded_signal = TensorHandle::<TestRuntime>::new_contiguous(
        shape,
        client.create_from_slice(f32::as_bytes(&padded_data)),
        dtype,
    );

    let virtual_re = empty_tensor(&client, vec![1, n_freq], dtype);
    let virtual_im = empty_tensor(&client, vec![1, n_freq], dtype);
    let padded_re = empty_tensor(&client, vec![1, n_freq], dtype);
    let padded_im = empty_tensor(&client, vec![1, n_freq], dtype);

    rfft_launch_padded::<TestRuntime>(
        &client,
        virtual_signal.binding(),
        virtual_re.clone().binding(),
        virtual_im.clone().binding(),
        1,
        signal_len,
        dtype,
    )
    .unwrap();
    rfft_launch_padded::<TestRuntime>(
        &client,
        padded_signal.binding(),
        padded_re.clone().binding(),
        padded_im.clone().binding(),
        1,
        n_fft,
        dtype,
    )
    .unwrap();

    let actual_re = to_f32(HostData::from_tensor_handle(&client, virtual_re, HostDataType::F32));
    let actual_im = to_f32(HostData::from_tensor_handle(&client, virtual_im, HostDataType::F32));
    let expected_re = to_f32(HostData::from_tensor_handle(&client, padded_re, HostDataType::F32));
    let expected_im = to_f32(HostData::from_tensor_handle(&client, padded_im, HostDataType::F32));

    for (actual, expected) in actual_re.iter().zip(expected_re.iter()) {
        assert!((actual - expected).abs() < 1e-4);
    }
    for (actual, expected) in actual_im.iter().zip(expected_im.iter()) {
        assert!((actual - expected).abs() < 1e-4);
    }
}

#[test]
fn irfft_spec_bins_matches_materialized_zero_padding() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let n_fft = 8usize;
    let n_freq = n_fft / 2 + 1;
    let spec_bins = 3usize;

    let virtual_re_data = vec![1.0, 0.5, -0.25, 4.0, -2.0];
    let virtual_im_data = vec![0.0, -0.125, 0.75, 8.0, 6.0];
    let mut padded_re_data = virtual_re_data.clone();
    let mut padded_im_data = virtual_im_data.clone();
    for value in padded_re_data.iter_mut().take(n_freq).skip(spec_bins) {
        *value = 0.0;
    }
    for value in padded_im_data.iter_mut().take(n_freq).skip(spec_bins) {
        *value = 0.0;
    }

    let virtual_re_in = TensorHandle::<TestRuntime>::new_contiguous(
        vec![1, n_freq],
        client.create_from_slice(f32::as_bytes(&virtual_re_data)),
        dtype,
    );
    let virtual_im_in = TensorHandle::<TestRuntime>::new_contiguous(
        vec![1, n_freq],
        client.create_from_slice(f32::as_bytes(&virtual_im_data)),
        dtype,
    );
    let padded_re_in = TensorHandle::<TestRuntime>::new_contiguous(
        vec![1, n_freq],
        client.create_from_slice(f32::as_bytes(&padded_re_data)),
        dtype,
    );
    let padded_im_in = TensorHandle::<TestRuntime>::new_contiguous(
        vec![1, n_freq],
        client.create_from_slice(f32::as_bytes(&padded_im_data)),
        dtype,
    );

    let virtual_signal = empty_tensor(&client, vec![1, n_fft], dtype);
    let padded_signal = empty_tensor(&client, vec![1, n_fft], dtype);

    irfft_launch_padded::<TestRuntime>(
        &client,
        virtual_re_in.binding(),
        virtual_im_in.binding(),
        virtual_signal.clone().binding(),
        1,
        spec_bins,
        dtype,
    )
    .unwrap();
    irfft_launch_padded::<TestRuntime>(
        &client,
        padded_re_in.binding(),
        padded_im_in.binding(),
        padded_signal.clone().binding(),
        1,
        n_freq,
        dtype,
    )
    .unwrap();

    let actual = to_f32(HostData::from_tensor_handle(
        &client,
        virtual_signal,
        HostDataType::F32,
    ));
    let expected = to_f32(HostData::from_tensor_handle(
        &client,
        padded_signal,
        HostDataType::F32,
    ));

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-4);
    }
}
