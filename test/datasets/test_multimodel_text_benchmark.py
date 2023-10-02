from torch_frame.datasets import MultimodalTextBenchmark


def test_multimodal_text_benchmark():
    dataset = MultimodalTextBenchmark('.', 'mercari_price_suggestion100K')
    breakpoint()
