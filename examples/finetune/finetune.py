from llmtoolkit import (
    get_args,
    get_accelerate_model,
    build_data_module,
    get_unique_key,
    train,
)

model_args, data_args, training_args = get_args()

model, tokenizer = get_accelerate_model(model_args, training_args)

data_module = build_data_module(tokenizer, "gsm8k")

train(
    model,
    tokenizer,
    data_module["train_dataset"],
    data_module["eval_dataset"],
    data_module["data_collator"],
    training_args,
    get_unique_key(model_args, data_args, training_args),
)