from datasets import load_dataset
from data.buffer import TokenBuffer

# TODO : move this to data folder
def get_buffer(
    model,
    batch_size,
    device,
    ctx_len,
):
    dataset = load_dataset(
        "wikipedia",
        language="en",
        date="20240401",
        split="train",
        streaming=True,
        trust_remote_code=True
    ).shuffle()
    dataset = iter(dataset)

    buffer = TokenBuffer(
        dataset,
        model,
        n_ctxs=10,
        ctx_len=ctx_len,
        load_buffer_batch_size=10,
        return_batch_size=batch_size,
        device=device,
        max_number_of_yields=2**20,
        discard_bos=True
    )

    return buffer