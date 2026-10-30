---
name: tensorlings-detailed-curriculum
kind: doc
description: the 13 sections expanded into chapters, exercises and projects, about 340 exercises, with the projects the table left open
updated: 2026-10-30
links: [docs/research/tensorlings-course-design.md]
---

# tensorlings detailed curriculum

This expands the 13 sections of [the course design](tensorlings-course-design.md#the-curriculum) into chapters, exercises and projects. The section table there is the one the owner approved, and this list fills it in without moving a topic between sections. The owner asked for it on 2026-10-29 and left the projects the table does not name to the agent, so each one below comes from its own section's topics and is modeled on a known course.

It plans about 340 exercises. Names and counts can still shift while a phase is built, and once a phase lands the built course is the truth. The owner's review of 2026-10-29 added 11 exercises and the 38_soft_design chapter, per [DEC-017](../../decisions/DEC-017-the-owner-s-six-gaps-added-to-the-curriculum.md), which moved every chapter after it up one or two numbers.

## How to read it

- A chapter is a folder in `exercises/`, numbered in course order. The numbers carry on from phase 1a, so `01_tensors` and `02_matmul` keep their names.
- Quizzes go in `exercises/quizzes/` as in rustlings, and the tables list each one where it falls.
- A project is a chapter of two to four exercises at the end of its section. A project that trains for minutes has the learner run the file, which saves a checkpoint to `checkpoints/`, and the check loads it and measures the result.
- From Section 3 on, a block gets the five exercises of [how every block is taught](tensorlings-course-design.md#how-every-block-is-taught) where each makes sense, so a block with no paper gets no paper exercise.
- Training that gains from a GPU runs on the best device the machine has, CUDA, MPS or the CPU, and short exercises run on the CPU, per [DEC-016](../../decisions/DEC-016-training-picks-the-best-device-and-exercises-stay-on-the-cpu.md).
- Datasets download on first use into `data/`, which git ignores. The owner allowed names.txt, tiny Shakespeare, MNIST and CIFAR-10 on 2026-10-29.

## Section 1 Foundations

| Chapter | Exercise | What it teaches |
|---|---|---|
| 01_tensors | tensors1 to reductions1 | built in phase 1a |
| 02_matmul | matmul1 | built in phase 1a |
| quizzes | quiz1 | built in phase 1a |
| 03_derivatives | derivatives1 | the slope at a point, estimated by nudging the input a little |
| | derivatives2 | the power, sum and product rules, each checked against the estimate |
| | derivatives3 | partial derivatives, nudging one input at a time, and the gradient as the list of them |
| 04_chain_rule | chain1 | the derivative of f(g(x)) as the product of the local derivatives |
| | chain2 | a value used twice, where the gradients from each path add up |
| | chain3 | backprop by hand through a small expression, node by node, as in Karpathy's first lecture |
| 05_autograd | autograd1 | a Value class that remembers how each number was made |
| | autograd2 | each operation's own backward step, with `=` where `+=` belongs as the error to fix |
| | autograd3 | backward through the whole expression in topological order |
| | autograd4 | PyTorch's autograd, `requires_grad`, `.backward()` and `.grad`, and gradients that pile up until zeroed |
| 06_tensor_backprop | backprop1 | gradients of elementwise operations and sums, which have the shape of the tensor they belong to |
| | backprop2 | the gradient through broadcasting, summed over the copied dimension |
| | backprop3 | the gradient of matrix multiplication, found from the shapes |
| | backprop4 | backward by hand through a few tensor steps, checked against PyTorch |
| quizzes | quiz2 | derivatives, the chain rule and backprop together |
| 07_micrograd | micrograd1 | the project starts, subtraction, negation, powers and division in your engine, built from what it has |
| | micrograd2 | exp and log with their backward steps |
| | micrograd3 | a gradient check of your engine against numerical estimates and PyTorch on random expressions |

The project finishes the engine of `05_autograd`, as Karpathy's micrograd does, without its neural net part, which belongs to Section 3. It beat a hand backprop of one big tensor expression and a gradient checker on its own, since the engine covers both and the check is its last step.

## Section 2 Learning

| Chapter | Exercise | What it teaches |
|---|---|---|
| 08_linear_regression | linreg1 | a model as a function with parameters, the prediction `x @ w + b`, with features and the bias named |
| | linreg2 | the mean squared error, the first loss, measuring how far predictions land from the targets |
| | linreg3 | the exact best fit from least squares, the answer gradient descent should reach |
| 09_gradient_descent | descent1 | one step downhill, the weights minus the learning rate times your hand gradient of the loss |
| | descent2 | a training loop on PyTorch's autograd, the update under `torch.no_grad()`, with a forgotten zeroing to fix |
| | descent3 | learning rates that crawl, converge and blow up |
| | descent4 | features on very different scales, standardized with the training data's mean and spread |
| 10_minibatches | minibatch1 | the gradient of a random minibatch as a noisy estimate of the full one |
| | minibatch2 | shuffling, slicing into minibatches, and the epoch |
| | minibatch3 | PyTorch's `TensorDataset` and `DataLoader` matched against your own batches |
| 11_softmax_regression | softmax1 | scores to probabilities with softmax, and the logit |
| | softmax2 | big scores that overflow, fixed by taking the largest away first |
| | softmax3 | softmax regression, one column of weights per class, the predicted class and the accuracy |
| 12_likelihood | likelihood1 | the likelihood of the labels, and why many probabilities multiplied reach zero while their logs add fine |
| | likelihood2 | the negative log likelihood is cross entropy, built from logits and matched with `F.cross_entropy` |
| | likelihood3 | the mean squared error is the negative log likelihood under Gaussian noise |
| | likelihood4 | the gradient of cross entropy with respect to the logits by hand, the probabilities minus the one hot labels |
| 13_generalization | splits1 | train, validation and test sets, cut after a shuffle, with a cut of sorted data to fix |
| | overfitting1 | polynomial features of a noisy curve, where training loss keeps falling and validation loss turns up |
| | overfitting2 | early stopping, keeping the weights from the step with the best validation loss |
| | decay1 | weight decay, the squared size of the weights added to the loss, and the pull toward zero it adds to each step |
| | decay2 | choosing the decay strength on the validation set, then measuring once on the test set |
| quizzes | quiz3 | Section 2 together |
| 14_mnist | mnist1 | the project starts, the MNIST digits loaded, flattened and scaled, with a validation set split off |
| | mnist2 | softmax regression trained with minibatches and weight decay past 90 percent validation accuracy |
| | mnist3 | the decay picked on validation, test accuracy measured once, and the digit pairs it confuses most |

The project is CS231n's first assignment, a softmax classifier on real images with its strength of regularization tuned on validation. It beat the same pipeline on generated data, whose result means less, and curve fitting, which `13_generalization` already covers.

## Section 3 Building blocks

| Chapter | Exercise | What it teaches |
|---|---|---|
| 15_linear | modules1 | `nn.Module` and `nn.Parameter`, how PyTorch holds the parameters of a model |
| | linear1 | what Linear computes, `x @ W.T + b`, with its weight shape of outputs by inputs |
| | linear2 | Linear built as a module and matched with `nn.Linear` given the same weights |
| | linear3 | two Linears in a row collapse into one, shown by multiplying their weights |
| 16_activations | relu1 | ReLU and its derivative, built and matched with `F.relu` |
| | relu2 | why, a task a Linear model can't fit and Linear, ReLU, Linear can |
| | activations1 | sigmoid and tanh, outputs held between bounds, and the slope that vanishes at large inputs |
| | gelu1 | GELU from its paper's formula and its tanh approximation, matched with `F.gelu` |
| | activations2 | when, the activation each job needs, a hidden feature, a gate, a bounded value or a choice among options |
| | soft1 | why `if`, `argmax` and indexing pass no gradient, and sigmoid and softmax as their soft versions |
| | soft2 | a soft lookup, softmax over how well a query matches each key, weighting the values |
| 17_mlp | mlp1 | Linear and activation stacked into an MLP with `nn.Sequential`, and its parameter count |
| | mlp2 | an MLP module built from a list of layer sizes and matched with `nn.Sequential` |
| | mlp3 | a curve fitted by an MLP with a full training loop over `model.parameters()` |
| | mlp4 | width and depth, where too small underfits and too big overfits |
| 18_embeddings | embedding1 | a one hot vector times a matrix picks a row, and an embedding is that lookup |
| | embedding2 | Embedding built and matched with `nn.Embedding` |
| | embedding3 | why, ids fed in as plain numbers mislead a model and an embedding fixes it |
| | embedding4 | discrete inputs in through an embedding, and a Linear head with softmax for the output |
| 19_initialization | init1 | how the spread of activations grows or shrinks through a deep stack |
| | init2 | the Xavier and Kaiming scales from their papers, matched with `nn.init` |
| | init3 | why, a deep MLP that stalls with bad scales and trains with Kaiming, and the starting loss, which should be the log of the class count |
| | init4 | when, the initialization each activation needs |
| 20_dropout | dropout1 | dropout, zeroing with probability p and scaling the rest, in training and eval mode |
| | dropout2 | why, an MLP that overfits a small set, with and without dropout |
| | dropout3 | the paper's scaling at test time next to the inverted dropout PyTorch uses, and where dropout goes in a model |
| 21_normalization | batchnorm1 | BatchNorm from its paper, built and matched with `nn.BatchNorm1d` in training mode |
| | batchnorm2 | running statistics for eval mode, with a forgotten `model.eval()` to fix |
| | layernorm1 | LayerNorm, each example over its own features, matched with `nn.LayerNorm` |
| | rmsnorm1 | RMSNorm, dividing by the root mean square only, matched with `nn.RMSNorm` |
| | norm1 | why and when, a deep MLP with and without normalization, BatchNorm for images and LayerNorm or RMSNorm for sequences |
| 22_residuals | residual1 | `x + f(x)`, and the gradient that passes the addition untouched |
| | residual2 | why, a deep plain MLP that won't train next to the same one with residuals |
| | residual3 | the pre norm residual block, which starts as "do nothing" with its last layer at zero |
| | residual4 | when, the order of norm, sublayer and residual in the blocks of the papers |
| quizzes | quiz4, quiz5 | the blocks together |
| 23_names | names1 | the project starts, the names dataset, a character vocabulary, and the counting bigram model as the loss to beat |
| | names2 | the MLP language model of Bengio and colleagues, with embeddings, normalization and a context of three characters |
| | names3 | the model trained below the bigram's validation loss, then new names sampled from it |

The project is the name generator of the section table, as in Karpathy's makemore parts 1 to 3.

## Section 4 Optimization

| Chapter | Exercise | What it teaches |
|---|---|---|
| 24_sgd | sgd1 | the optimizer object, SGD built as a class with `step` and `zero_grad`, matched with `torch.optim.SGD` |
| | momentum1 | momentum, a running velocity, matched with `SGD(momentum=0.9)` |
| | momentum2 | why, a long narrow valley where SGD zigzags and momentum runs along it |
| 25_adaptive | rmsprop1 | RMSProp, each step divided by a running root mean square of the gradient, matched with `torch.optim.RMSprop` |
| | adam1 | Adam from Algorithm 1 of its paper, matched with `torch.optim.Adam` |
| | adam2 | why the bias correction, the oversized early steps without it |
| | adamw1 | decoupled weight decay from the AdamW paper against L2 added to the gradient, matched with `torch.optim.AdamW` |
| | adamw2 | when, which optimizer and learning rate range fits which model |
| 26_schedules | schedule1 | step and cosine decay as functions of the step, matched with `torch.optim.lr_scheduler` |
| | warmup1 | linear warmup, and a deep model whose first steps blow up without it |
| | schedule2 | a learning rate range test, sweeping the rate and picking one before the loss blows up |
| 27_clipping | clipping1 | the global gradient norm and clipping it, matched with `clip_grad_norm_` |
| | clipping2 | why, one huge gradient that throws training off, held in check by clipping |
| 28_debugging | debug1 | Karpathy's recipe, starting with the loss at step zero and a model whose start is off |
| | debug2 | overfitting one batch, and the bug in a model that can't |
| | debug3 | gradient norms per layer, and the layer that gets none |
| | debug4 | looking at the data, and inputs and labels that no longer line up |
| | debug5 | the common slips, a missing `zero_grad`, `train` and `eval` modes, and a summed loss where a mean was meant |
| quizzes | quiz6 | Section 4 together |
| 29_broken_training | broken1 | the project starts, the Section 3 name generator with planted bugs, found with the recipe one step at a time |
| | broken2 | the fixed run tuned with AdamW, warmup and clipping below a validation loss target |

The project comes from the section's last topic, the recipe, since a broken run calls on the whole section. It is modeled on Karpathy's recipe post and CS231n's babysitting of the learning process.

## Section 5 Convolutional nets

| Chapter | Exercise | What it teaches |
|---|---|---|
| 30_convolution | conv1 | 1D convolution as a Linear with most weights zero and the rest shared |
| | conv2 | 1D convolution as a sliding dot product, then with `unfold`, matched with `F.conv1d` |
| | conv3 | 2D convolution on a small image, matched with `F.conv2d` |
| | conv4 | why, a Linear on pixels can't spot a pattern where it never saw it and a convolution can |
| | fft1 | the convolution theorem, a convolution as a product of spectra with `torch.fft.rfft`, matched with `F.conv1d` |
| 31_conv_shapes | channels1 | input and output channels, the weight shape, and the parameter count against a Linear |
| | padding1 | padding that keeps the size, and the output size formula |
| | stride1 | stride, which shrinks the output |
| | dilation1 | dilation, a wider view with the same weights, and the receptive field |
| | conv5 | when, the 1 by 1 convolution, and kernel, stride and padding chosen for a target shape |
| 32_pooling | pooling1 | max and average pooling, matched with `F.max_pool2d` |
| | pooling2 | why, pooling after a shared weight layer turns "the same response anywhere" into one answer |
| 33_classic_nets | lenet1 | LeNet-5 from its paper, the shapes layer by layer, trained on MNIST |
| | vgg1 | stacks of 3 by 3 convolutions, which see what one 5 by 5 sees with fewer weights |
| | resnet1 | the ResNet basic block with BatchNorm and the projection shortcut |
| | resnet2 | why, a deeper plain net that trains worse than a shallow one, and the ResNet that doesn't |
| 34_upsampling | upsample1 | nearest and bilinear upsampling, matched with `F.interpolate` |
| | transposed1 | transposed convolution as the backward pass of a convolution, matched with `nn.ConvTranspose2d` |
| | unet1 | U-Net from its paper, the way down, the way up and the skip connections |
| | unet2 | why the skips, segmenting made up shapes with and without them |
| 35_augmentation | augment1 | flips, padded crops and shifts as tensor operations |
| | augment2 | why and when, a small training set with and without augmentation, and changes that would break the label |
| 36_transfer | transfer1 | loading weights, freezing the body and training a new head |
| | transfer2 | why, a hundred labeled images where a body trained on another task beats training from scratch |
| quizzes | quiz7 | Section 5 together |
| 37_cifar | cifar1 | the project starts, CIFAR-10 loaded and normalized with a validation set split off |
| | cifar2 | a small ResNet with augmentation, trained by running the file, past 80 percent on validation |
| | cifar3 | test accuracy measured once, and the classes it confuses |

The project is the small image classifier of the section table, in the style of CS231n's convolutional net assignment.

## Section 6 Sequences

| Chapter | Exercise | What it teaches |
|---|---|---|
| 38_soft_design | derive1 | the design loop on a memory that keeps the last marked value, the exact algorithm by hand, its hard step found, and a sigmoid gate in its place |
| | derive2 | the same loop on a lookup by key, ending in a softmax over how well each key matches |
| 39_rnn | rnn1 | one Linear reused at every step with a hidden state, matched with `nn.RNN` |
| | rnn2 | batches of sequences and their shapes |
| | rnn3 | why, a task that needs the whole past, which a fixed window can't hold and an RNN can |
| 40_bptt | bptt1 | backprop through time by hand, where repeated multiplying shrinks or grows the gradient |
| | bptt2 | truncated backprop through time, detaching the hidden state between chunks |
| | bptt3 | gradients that vanish over long spans, and clipping for the ones that explode |
| 41_gated_rnns | lstm1 | the LSTM cell from its papers, forget gate included, matched with `nn.LSTM` |
| | lstm2 | why, remembering one value from 50 steps back, where the RNN fails and the LSTM doesn't |
| | gru1 | the GRU, matched with `nn.GRU` |
| | gating1 | a gate, one signal scaling another through a sigmoid, the first weight here that is computed from the input instead of fixed by training |
| | lstm3 | when, an RNN, an LSTM or a GRU |
| 42_bidirectional | bidir1 | one RNN forwards and one backwards, matched with `bidirectional=True` |
| | bidir2 | when, tagging can read ahead and generating can't |
| 43_seq2seq | encdec1 | an encoder into one vector and a decoder that writes step by step with teacher forcing |
| | encdec2 | why one vector struggles with long inputs |
| | greedy1 | greedy decoding until the end token |
| | beam1 | beam search over total log probability, with length normalization |
| 44_causal_conv | causal1 | causal convolution, padded on the left so no output sees the future |
| | dilated1 | dilated causal stacks from WaveNet, whose view doubles each layer |
| | wavenet1 | why, a long view in few layers that trains in parallel |
| | stft1 | the short time Fourier transform with `torch.stft`, overlapping windows of a signal and their spectra, the form most audio models read |
| 45_state_space | ssm1 | a linear state space recurrence, run as a loop and as a convolution |
| | ssm2 | a diagonal state, discretization and the condition for stability |
| | ssm3 | Mamba's selective step that depends on the input, and a selective copying task the fixed version fails |
| quizzes | quiz8 | Section 6 together |
| 46_adder | adder1 | the project starts, sums written as text like `123+456`, and their tokens |
| | adder2 | a GRU encoder and decoder that gets 95 percent of held out sums exactly right, with answers written units first, which beats the usual order by far |
| | adder3 | beam search against greedy decoding, and longer numbers than training had, where it breaks |

The project follows Zaremba and Sutskever's learning to execute and the Keras addition RNN. It uses an encoder and decoder, gating and beam search at once.

## Section 7 Attention and transformers

| Chapter | Exercise | What it teaches |
|---|---|---|
| 47_attention | attention1 | each position as the average of the ones before it, with a loop and then a triangular matrix |
| | attention2 | weighted averages, softmax over scores with the future masked out |
| | attention3 | scores from the data, queries and keys, scaled by the square root of their size |
| | attention4 | values and the output layer, and why attention projects before comparing and merges after |
| | attention5 | scaled dot product attention built and matched with `F.scaled_dot_product_attention` |
| | attention6 | why, recalling the value stored with a key many steps back, where an RNN's fixed vector fails and a query compared with every key succeeds |
| 48_linear_attention | linattn1 | softmax dropped, so (Q K^T) V becomes Q (K^T V), a recurrence whose state is a matrix of keys times values |
| | linattn2 | when, a fixed size state against attention's growing one, on recall of more pairs than the state can hold |
| 49_masks | mask1 | causal and padding masks, and both at once |
| | mask2 | why, a model without the causal mask that cheats in training and fails at generation |
| 50_multihead | multihead1 | splitting into heads with reshape and transpose |
| | multihead2 | multi head attention built and matched with `nn.MultiheadAttention` |
| | multihead3 | why heads, a task that needs two lookups at once |
| | gqa1 | grouped query attention, fewer key and value heads shared, and the memory it saves |
| 51_cross_attention | cross1 | queries from one sequence, keys and values from another |
| | cross2 | why, copying long rows of digits with cross attention in place of one vector, since long sums trained in neither form within minutes |
| 52_positions | position1 | attention alone ignores order, shown by shuffling the input |
| | position2 | learned and sinusoidal positions |
| | rope1 | RoPE from its paper, pairs of features rotated by position |
| | rope2 | why, RoPE scores that depend only on the distance between positions |
| 53_transformer_block | block1 | the block, pre norm attention and a pre norm MLP, each with a residual, mixing across positions then within each |
| | block2 | the GPT-2 block with LayerNorm and a GELU MLP four times as wide |
| | block3 | why each part, removing one at a time on a toy task |
| | induction1 | why stacking, copying the token that followed an earlier match, which one attention layer can't do and a shift then attention can |
| 54_model_types | encoder1 | encoder only, decoder only and encoder decoder models, and the masks that set them apart |
| | decoder1 | a decoder only language model with shifted targets, end to end |
| 55_modern_llms | llama1 | GPT-2 to Llama, RMSNorm in place of LayerNorm and RoPE in place of learned positions |
| | llama2 | the SwiGLU MLP from its paper, and why its width is two thirds of four times the model size |
| | llama3 | a Llama block next to a GPT-2 block parameter by parameter, with the QK norm of recent models |
| 56_moe | moe1 | experts and a router with top k routing, running only the chosen experts |
| | moe2 | why the balancing loss, a router that collapses onto one expert without it |
| | moe3 | when, more parameters for the same compute |
| | dynamic1 | gates, attention, SwiGLU and routing as one idea, weights computed from the input on every forward pass instead of fixed by training |
| 57_vit | vit1 | an image as a sequence of patches, a patch embedding and a class token |
| | vit2 | a small ViT on MNIST next to a CNN, and how much data each needs |
| 58_bpe | bpe1 | text as UTF-8 bytes, the 256 starting tokens |
| | bpe2 | counting pairs and merging the most common one |
| | bpe3 | training merges, then encoding and decoding with a round trip |
| | bpe4 | why, characters make sequences long and words make the vocabulary huge, and the quirks of numbers and spaces |
| quizzes | quiz9, quiz10 | attention and transformers together |
| 59_gpt | gpt1 | the project starts, tiny Shakespeare in tokens, a train and validation split, and batches of blocks |
| | gpt2 | your blocks assembled into a GPT with GPT-2's scaled residual initialization |
| | gpt3 | training by running the file with AdamW, warmup, cosine decay and clipping, below a validation loss target |
| | gpt4 | generating text with temperature and top k |

The project is the GPT on real text of the section table, as in Karpathy's GPT lecture and nanoGPT.

## Section 8 Representation learning

| Chapter | Exercise | What it teaches |
|---|---|---|
| 60_word2vec | word2vec1 | skip gram pairs from a window of text |
| | word2vec2 | skip gram with negative sampling, trained on the words of tiny Shakespeare |
| | word2vec3 | nearest words by cosine similarity, and why one hot vectors have no notion of near |
| 61_contrastive | contrastive1 | the similarity matrix of two views, and the InfoNCE loss as cross entropy over the batch |
| | contrastive2 | two augmented views of one image as a pair, and the temperature |
| | contrastive3 | why, features learned with no labels, and the collapse when nothing pushes apart |
| 62_masked_modeling | masked1 | BERT's masking of random tokens, with the loss only on masked positions |
| | masked2 | MAE's masking of most image patches, and why images need a high ratio |
| 63_autoencoders | autoencoder1 | an encoder into a small code and a decoder back, with a reconstruction loss |
| | autoencoder2 | why the bottleneck matters, and the denoising autoencoder |
| | autoencoder3 | a linear autoencoder finds what PCA finds, checked against `torch.pca_lowrank` |
| quizzes | quiz11 | Section 8 together |
| 64_probe | probe1 | the project starts, SimCLR style training on CIFAR-10 images with no labels |
| | probe2 | a linear probe on the frozen features with a few labels |
| | probe3 | the probe against random features and a supervised model given the same labels |

The project comes from contrastive learning, since its features can be measured, and follows SimCLR's linear evaluation.

## Section 9 Generative models

| Chapter | Exercise | What it teaches |
|---|---|---|
| 65_sampling | sampling1 | drawing from a categorical distribution, with `torch.multinomial` and by hand |
| | sampling2 | temperature, top k and top p on logits |
| | sampling3 | the reparameterization trick, a Gaussian sample as the mean plus the spread times noise |
| 66_vae | vae1 | an encoder that outputs a mean and a log variance, and the KL term in closed form |
| | vae2 | the ELBO, a small VAE on MNIST, and new digits sampled from it |
| | vae3 | why, the holes in an autoencoder's code space against the VAE's smooth one |
| 67_gan | gan1 | generator and discriminator and their two losses from the paper |
| | gan2 | the alternating loop on a ring of 2D points, and mode collapse |
| | gan3 | when, sharp samples but unstable training, with a missing `detach` to fix |
| 68_flows | flow1 | change of variables, the log density plus the log determinant |
| | flow2 | an affine coupling layer from RealNVP that inverts exactly |
| | flow3 | a flow trained on two moons by maximum likelihood, then sampled |
| 69_diffusion | diffusion1 | the noising process and its closed form from the DDPM paper |
| | diffusion2 | the loss, predicting the noise |
| | diffusion3 | the DDPM sampling loop, then DDIM with fewer steps |
| | diffusion4 | conditioning with classifier free guidance |
| 70_vqvae | vq1 | the nearest codebook entry and the straight through gradient |
| | vq2 | codebook and commitment losses, and images as tokens |
| quizzes | quiz12 | Section 9 together |
| 71_digit_diffusion | digits1 | the project starts, a small U-Net denoiser for MNIST that takes the digit as a condition |
| | digits2 | training by running the file |
| | digits3 | sampling with guidance, scored by your own MNIST classifier on whether each sample is the digit asked for |

The project is diffusion, the method most image generators use, on the MNIST data the course already has.

## Section 10 Symmetry and graphs

| Chapter | Exercise | What it teaches |
|---|---|---|
| 72_sets | sets1 | Deep Sets, features per element pooled by a sum, unchanged by shuffling |
| | sets2 | why, an MLP on a fixed order that fails on shuffled sets |
| | sets3 | equivariant layers, and attention with no positions as a set layer |
| 73_graphs | graph1 | a graph as an adjacency matrix and as an edge list |
| | message1 | message passing from Gilmer and colleagues, with `index_add_` |
| | message2 | the GCN's normalization from its paper, and oversmoothing as depth grows |
| | message3 | why, message passing lines up with the steps of Bellman-Ford |
| 74_equivariance | equivariance1 | f(g x) equals g f(x), tested for shifts, shuffles and rotations |
| | equivariance2 | rotation invariant features from distances, after the E(n) equivariant GNN |
| | equivariance3 | when, a symmetry built in against a symmetry taught by augmentation |
| | equivariance4 | the structure chosen from the data's symmetries, the Geometric Deep Learning blueprint |
| quizzes | quiz13 | Section 10 together |
| 75_shortest_paths | paths1 | the project starts, random graphs and their shortest path distances |
| | paths2 | a message passing network against an MLP on the adjacency matrix |
| | paths3 | both on bigger graphs than training had |

The project follows Xu and colleagues' algorithmic alignment. It shows structure chosen from the problem beating a bigger generic model.

## Section 11 Post training and RL

| Chapter | Exercise | What it teaches |
|---|---|---|
| 76_finetuning | finetune1 | your Section 7 GPT fine tuned on new text with a small learning rate, and what it forgets |
| | finetune2 | freezing layers, and which ones to freeze |
| 77_lora | lora1 | LoRA from its paper, a low rank update that starts at zero |
| | lora2 | the parameters it saves, and merging the update back |
| | lora3 | why, close to full fine tuning with a hundredth of the parameters |
| 78_sft | sft1 | a chat format with special tokens, and a loss on the answer tokens only |
| | sft2 | the GPT taught a question and answer format |
| 79_policy_gradients | rl1 | policy, action, reward and episode on a bandit |
| | reinforce1 | REINFORCE, the log probability times the reward |
| | reinforce2 | a baseline that cuts the noise, and the advantage |
| | reinforce3 | why, a score with no gradient can still train a policy |
| 80_ppo | ppo1 | the probability ratio and PPO's clipped objective |
| | ppo2 | generalized advantage estimation |
| | ppo3 | a PPO loop on a small balancing task written in plain PyTorch |
| 81_reward_models | reward1 | a reward model trained on preference pairs with the Bradley Terry loss |
| | reward2 | the KL penalty to the reference model, and reward hacking on a toy |
| 82_dpo | dpo1 | the DPO loss from its paper |
| | dpo2 | the GPT trained with DPO, next to the reward model and PPO route |
| | grpo1 | GRPO, advantages taken relative to a group of samples with no value model |
| quizzes | quiz14 | Section 11 together |
| 83_post_training | post1 | the project starts, LoRA SFT of your GPT into an answer format on a made up task |
| | post2 | DPO toward the preferred answers |
| | post3 | the result scored by exact match and by your reward model |

The project works on the GPT the learner already trained, the way nanochat runs SFT and then RL on its own base model.

## Section 12 Systems and inference

| Chapter | Exercise | What it teaches |
|---|---|---|
| 84_cost | cost1 | the FLOPs of a matrix multiply, a Linear and attention, and the memory of the parameters |
| | cost2 | activation memory for the backward pass, and six FLOPs per parameter per token in training |
| 85_profiling | profile1 | timing done right with warmup and synchronizing, and `torch.profiler` |
| | profile2 | memory bound against compute bound, and why fusing elementwise steps helps |
| 86_precision | precision1 | float32, float16 and bfloat16, their range and their precision |
| | precision2 | autocast, and loss scaling for float16 |
| 87_flash_attention | flash1 | the online softmax, one pass with a running maximum and total |
| | flash2 | FlashAttention's tiling in plain PyTorch, blocks of keys and values with no full score matrix, matched with full attention |
| 88_kv_cache | kvcache1 | keys and values cached while generating, with the same output as without |
| | kvcache2 | why, the time per token with and without the cache, and the cache's memory |
| 89_quantization | quant1 | absmax int8 quantization and its error |
| | quant2 | scales per channel against one per tensor, and outliers |
| | quant3 | your GPT's Linears in int8, its size and its validation loss |
| 90_parallel | ddp1 | data parallel training by hand, where the average of the workers' gradients is the full batch gradient |
| | ddp2 | gradient accumulation, the same thing on one device |
| 91_scaling | scaling1 | a power law fitted in log space |
| | scaling2 | Chinchilla's compute optimal size for a budget |
| | scaling3 | your own small sweep of GPT sizes and its fitted curve |
| 92_data | data1 | exact and near duplicate removal |
| | data2 | quality filters, and checking the test set for leaked text |
| quizzes | quiz15 | Section 12 together |
| 93_fast_gpt | fast1 | the project starts, your GPT's generation profiled |
| | fast2 | the KV cache and int8 weights added |
| | fast3 | tokens per second and validation loss before and after |

The project lands the section's main tools on one model and measures them end to end. CUDA kernels, Triton and several GPUs can't run on the owner's Mac, so those ideas are taught in plain PyTorch, per [the limits](tensorlings-course-design.md#limits-on-this-machine).

## Section 13 Your own modules

| Chapter | Exercise | What it teaches |
|---|---|---|
| 94_combining | combine1 | FiLM from its paper, a side input that scales and shifts features, its scale and shift computed from that input |
| | combine2 | gating against adding, when one signal should control another, and the effective weights the input writes on every pass |
| | combine3 | a new branch that starts as "do nothing" against one that starts random |
| | combine4 | mixing across positions alternated with mixing within each, against either alone |
| | combine5 | picking the combination for problems you haven't seen |
| 95_design_method | design1 | a toy task that isolates one skill, associative recall as in Zoology |
| | design2 | the simplest baseline and the diagnosis, failed to fit or failed to generalize |
| | design3 | the design loop of 38_soft_design on a harder algorithm, with several hard steps to swap for differentiable ones |
| | design4 | gradient health before training, the identity at the start and gradients that reach every layer |
| | design5 | the property tested directly by shuffling, shifting, rotating or lengthening the input |
| 96_papers | paper1 | a paper's method turned into shapes and code, with the squeeze and excitation block |
| | paper2 | a result reproduced small, with its spread across seeds |
| 97_interpretability | interp1 | forward hooks, and probes on hidden states |
| | interp2 | attention patterns and induction heads in a small model trained on repeated sequences, the circuit induction1 built by hand |
| | interp3 | ablation and activation patching |
| | interp4 | attribution by gradients and by integrated gradients |
| quizzes | quiz16 | the whole course |
| 98_echo_canceller | echo1 | the project starts, a far end signal through a room's impulse response plus near end sound, seen in time and in STFT frames |
| | echo2 | the classic NLMS adaptive filter as the baseline, and why it converges slowly on speech |
| | echo3 | a learned module on STFT frames, where the room becomes a product per frequency, built with the design method |
| | echo4 | echo return loss enhancement against the baseline, and the cases each one wins |
| 99_quant_forecaster | forecast1 | the project starts, a made up price series with a weak signal of known size |
| | forecast2 | the leaks to find, future data in the features, statistics from the whole series and a random split |
| | forecast3 | walk forward validation with every leak fixed |
| | forecast4 | a small model against a plain baseline, and the honest result |
| 100_own_module | own1 | the project starts, a module of your choice planned with the design method |
| | own2 | its property tests, and its result against a baseline |

## What goes past the table

These are details the table does not name. Each sits inside its section's topic, and the owner can drop any of them.

- Standardizing features and early stopping in Section 2, since gradient descent and overfitting are hard to show honestly without them.
- Grouped query attention and QK norm in Section 7, part of the GPT-2 to Llama path in [Raschka's comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison), updated April 2026.
- GRPO in Section 11, the policy gradient method behind DeepSeek R1 and the RL stage of [nanochat](https://github.com/karpathy/nanochat/discussions/1).
- The online softmax in Section 12, the step FlashAttention's tiling rests on.
- The NLMS filter in the echo canceller, the classic method a learned one has to beat.

## Phases

| Phase | Sections | Exercises |
|---|---|---|
| 1a | Section 1 up to matmul1 | 7, built |
| 1b | the rest of Section 1, and Section 2 | 44, built |
| 2 | 3 and 4 | 61, the first 59 built, then soft1 and soft2 added |
| 3 | 5, 6 and 7 | 103 |
| 4 | 8, 9 and 10 | 52 |
| 5 | 11 and 12 | 47 |
| 6 | 13 | 27 |
