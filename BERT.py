import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine, pdist, squareform
torch.set_num_threads(8)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True,
                                  # Whether the model returns all hidden-states.
                                  )
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def BERTembed(text, debug=False):
    # text = "Here is the sentence I want embeddings for."
    marked_text = "[CLS] " + text + " [SEP]"

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Print out the tokens.
    # print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # print(segments_ids)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # print(model.eval())

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]
        # print(outputs.shape)
    if debug:
        print("Number of layers:", len(hidden_states),
              "  (initial embeddings + 12 BERT layers)")
        layer_i = 0

        print("Number of batches:", len(hidden_states[layer_i]))
        batch_i = 0

        print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
        token_i = 0

        print("Number of hidden units:",
              len(hidden_states[layer_i][batch_i][token_i]))

        print('Tensor shape for each layer: ', hidden_states[0].size())
    token_embeddings = torch.stack(hidden_states, dim=0)
    if debug:
        print(token_embeddings.size())
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    if debug:
        print(token_embeddings.size())
    token_embeddings = token_embeddings.permute(1, 0, 2)

    if debug:
        print(token_embeddings.size())
    fembed = []
    # for token in token_embeddings:
    token_vecs = hidden_states[-2][0]
    embed = torch.mean(token_vecs, dim=0)
    # fembed.append(embed.detach().numpy())


    # Calculate the average of all 22 token vectors.
    # sentence_embedding = torch.mean(token_vecs, dim=0)
    # print(sentence_embedding.size())
    # print(sentence_embedding)
    return embed


if __name__ == '__main__':
    # a = BERTembed('kidding')
    # b = BERTembed('joking')
    w_list = ['embedding']
    embedding = []
    for i in w_list:
        # embedding.append(BERTembed(i))
        embedding = BERTembed(i)
    # print(1- cosine(a,b))
    a = pdist(embedding, metric='cosine')
    print(squareform(1 - a))
