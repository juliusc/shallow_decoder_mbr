This contains some code for a quick experiment attempting to distill a 1-layer encoder-decoder transformer model from a normal sized transformer for NMT. The purpose is for speculative sampling; sequence generation is bottlenecked by memory bandwidth due to the size of the decoder, and speculative sampling uses a draft model to quickly generate tokens, while the base model verifies or rejects the tokens.

The choice to use a 1-layer model is inspired by [Kasai et al., 2020](https://arxiv.org/abs/2006.10369), who find that doubling encoder layers and using a single decoder layer works pretty well while generating tokens much more quickly.

This experiment is incomplete and considered not super promising for now, for a few reasons:
- For the size of model I trained on, the smaller model isn't really even faster, and my guess is due to the output layer / softmax dominating the runtime.
- The 1-layer model, while surprisingly good, struggled with longer sequences. This is due to me freezing the encoder and sharing it with the base model (arguably not a great idea).

If I pick this up again, I would use larger models and larger datasets, as well as tweak the training of the draft model a bit.

This code is not in a very robust and usable state. I've uploaded it for my own documentation purposes, but this helps you in any way, feel free to send me a message.