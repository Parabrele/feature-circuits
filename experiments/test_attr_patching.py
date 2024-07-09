import einops
import torch
import plotly.express as px

from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]

answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

clean_tokens = model.tokenizer(prompts, return_tensors="pt")["input_ids"]

corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]

answer_token_indices = torch.tensor(
    [
        [model.tokenizer(answers[i][j])["input_ids"][0] for j in range(2)]
        for i in range(len(answers))
    ]
)

def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

clean_logits = model.trace(clean_tokens, trace=False).logits.cpu()
corrupted_logits = model.trace(corrupted_tokens, trace=False).logits.cpu()

CLEAN_BASELINE = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {CLEAN_BASELINE:.4f}")

CORRUPTED_BASELINE = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {CORRUPTED_BASELINE:.4f}")

def ioi_metric(
    logits,
    answer_token_indices=answer_token_indices,
):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )

print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")


clean_out = []
corrupted_out = []
corrupted_grads = []

with model.trace() as tracer:

    with tracer.invoke(clean_tokens) as invoker_clean:

        for layer in model.transformer.h:
            attn_out = layer.attn.c_proj.input[0][0]
            clean_out.append(attn_out.save())

    with tracer.invoke(corrupted_tokens) as invoker_corrupted:

        for layer in model.transformer.h:
            attn_out = layer.attn.c_proj.input[0][0]
            corrupted_out.append(attn_out.save())
            corrupted_grads.append(attn_out.grad.save())

        logits = model.lm_head.output.save()
        # Our metric uses tensors saved on cpu, so we
        # need to move the logits to cpu.
        value = ioi_metric(logits.cpu())
        value.backward()


patching_results = []

for corrupted_grad, corrupted, clean, layer in zip(
    corrupted_grads, corrupted_out, clean_out, range(len(clean_out))
):

    residual_attr = einops.reduce(
        corrupted_grad.value[:,-1,:] * (clean.value[:,-1,:] - corrupted.value[:,-1,:]),
        "batch (head dim) -> head",
        "sum",
        head = 12,
        dim = 64,
    )

    patching_results.append(
        residual_attr.detach().cpu().numpy()
    )


fig = px.imshow(
    patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    title="Patching Over Attention Heads"
)

fig.update_layout(
    xaxis_title="Head",
    yaxis_title="Layer"
)

fig.show()

######
# Now mine
######

from utils.experiments_setup import load_model_and_modules
from connectivity.functional import ROI_activation

model, embed, resids, attns, mlps, submod_names = load_model_and_modules(
    device="cpu",
    unified=True,
    model_name="gpt2",
)

attr = ROI_activation(
    clean=clean_tokens,
    model=model,
    metric_fn=ioi_metric,
    attns=attns,
    mlps=mlps,
    patch=corrupted_tokens,
    ablation_fn=lambda x: x,
)
attr = attr['attr']
print(len(attr))