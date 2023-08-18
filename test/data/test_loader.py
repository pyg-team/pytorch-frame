from torch_frame.data import DataLoader, TensorFrame


def test_data_loader(get_fake_tensor_frame):
    tf = get_fake_tensor_frame(num_rows=10)

    loader = DataLoader(tf, batch_size=3)
    assert len(loader) == 4

    for i, batch in enumerate(loader):
        assert isinstance(batch, TensorFrame)
        if i + 1 < len(loader):
            assert len(batch) == 3
        else:
            assert len(batch) == 1
