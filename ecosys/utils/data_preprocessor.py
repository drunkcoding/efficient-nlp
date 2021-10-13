


def texts_tokenizer(tokenizer, texts, labels, batch_size):
    encoded_texts = tokenizer(
        texts, 
        padding = 'max_length', 
        truncation = True, 
        max_length=128, 
        return_tensors = 'pt'
    )
    dataset = Dataset(encoded_texts, labels)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size
    )

    return dataloader