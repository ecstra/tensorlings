---
name: tensorlings-research
kind: doc
description: what expert courses agree a deep learning expert must know, the teaching research, and the research on designing modules, behind tensorlings
updated: 2026-10-28
links: []
---

# tensorlings research

The research behind the tensorlings course, gathered on 2026-10-27. The design built from it is in [the course design](tensorlings-course-design.md). Two earlier versions of the course, AIlings, failed with the owner and the reasons are recorded at the end, so they are not repeated.

## What an expert needs, by survey

The curriculum comes from what these sources agree on, not from any single framing.

- Karpathy, [Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html) and [its repo](https://github.com/karpathy/nn-zero-to-hero): micrograd, a bigram model, an MLP language model, activations and BatchNorm, backprop by hand through tensors, a WaveNet style model, GPT, the tokenizer.
- Karpathy, [LLM101n syllabus](https://github.com/karpathy/LLM101n): bigram, micrograd, MLP, attention, transformer, tokenization, optimization with AdamW, device, precision, distributed training, datasets, KV cache, quantization, SFT and LoRA, RLHF, PPO and DPO, deployment, VQ-VAE and the diffusion transformer.
- [Stanford CS336](http://cs336.stanford.edu/spring2025/): tokenizer, transformer and optimizer from scratch, GPUs, Triton kernels, FlashAttention, parallelism, scaling laws, inference, evaluation, data filtering, SFT, RLHF and DPO, mixture of experts.
- [Stanford CS231n](https://cs231n.stanford.edu/schedule.html): linear classifiers, optimizers, backprop, CNNs, BatchNorm, ResNet, RNN and LSTM, attention, detection and segmentation, distributed training, self supervised learning, VAEs, GANs, diffusion.
- [Stanford CS224n](https://web.stanford.edu/class/cs224n/): word vectors, backprop, RNN language models, transformers, pretraining, post training, PEFT, RAG and agents, evaluation, tokenization, interpretability.
- [MIT 6.7960](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/pages/syllabus/): MLPs, CNNs, RNNs, graph networks, transformers, invariances, autodiff, generalization.
- [Dive into Deep Learning](https://d2l.ai/): linear and softmax regression, MLPs, dropout, initialization, custom layers, CNNs to ResNet and DenseNet, RNN, LSTM, GRU, encoder and decoder, beam search, attention, transformers, ViT, optimizers and schedules, multi GPU, transposed convolution, word2vec, BERT, GANs, reinforcement learning.
- [Understanding Deep Learning](https://penguinrandomhouse.com/books/730887/understanding-deep-learning-by-simon-j-d-prince): losses, fitting, initialization, regularization, CNNs, residual networks, transformers, graph networks, GANs, normalizing flows, VAEs, diffusion, reinforcement learning.
- [PyTorch torch.nn](https://docs.pytorch.org/docs/2.14/nn.html): the plain list of layers the field uses, linear, convolution, pooling, padding, activations, normalization (BatchNorm, GroupNorm, InstanceNorm, LayerNorm, RMSNorm), recurrent, transformer, dropout, embedding, losses, upsampling.
- [GPT-2 to Llama comparison](https://sebastianraschka.com/faq/docs/gpt-llama-qwen-gemma-comparison.html): current language models swap LayerNorm for RMSNorm, learned positions for RoPE, the GELU MLP for SwiGLU, and full attention for grouped query attention.

Every serious syllabus covers six things together, data, the model, the loss, the optimizer, the systems it runs on, and evaluation. Architecture is one of the six, which is why the course covers all of them.

## The frame that failed

The first two versions framed every module as a constrained MLP, a frame taken from a chat the owner pasted. It holds for some layers. Convolution is a Linear layer with most weights removed and the rest shared, and an RNN reuses one layer at every step. It fails for others.

- Attention, gates and FiLM multiply two signals together, which an MLP only approximates ([Jayakumar and colleagues 2020](https://mlanthology.org/iclr/2020/jayakumar2020iclr-multiplicative)).
- Normalizing flows need layers that run backwards exactly.
- GANs, VAEs and diffusion are mostly a training objective around ordinary layers.
- Reinforcement learning changes where the learning signal comes from.

Other corrections to that chat, checked against sources.

- The forget gate was added to the LSTM in 2000 so it can clear its memory on long input streams ([Gers and colleagues](https://mlanthology.org/neco/2000/gers2000neco-learning)). It is not only about variance.
- Rewriting (QKᵀ)V as Q(KᵀV) is linear attention ([Katharopoulos and colleagues 2020](https://nebius.com/blog/posts/mixers-are-rnns)). Mamba is a recurrence whose update depends on the input ([paper](https://arxiv.org/pdf/2312.00752)), later shown to resemble linear attention with gates ([analysis](https://arxiv.org/pdf/2405.16605)).

## Research on combining and designing modules

- Programming constructs made differentiable ([The Elements of Differentiable Programming](https://arxiv.org/abs/2403.14606)): an if becomes a gate, an index becomes softmax addressing, a loop becomes recurrence, a switch becomes top k routing. Hard steps can also be relaxed with [Gumbel softmax and straight through estimators](https://arxiv.org/html/2410.13331v1) or [NeuralSort](https://www.emergentmind.com/papers/1903.08850.md).
- Token mixing alternates with channel mixing, and that alternation matters more than which token mixer is used ([MetaFormer](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_MetaFormer_Is_Actually_What_You_Need_for_Vision_CVPR_2022_paper.html)).
- LayerNorm before each sublayer keeps gradients healthy at the start ([Xiong and colleagues 2020](https://arxiv.org/pdf/2002.04745)). A residual branch scaled from zero starts the network as the identity ([ReZero](https://arxiv.org/abs/2003.04887)).
- Gates and residuals share a line of descent, from LSTM gates to [Highway networks](https://proceedings.neurips.cc/paper_files/paper/2015/file/215a71a12769b056c3c32e7299f1c5ed-Paper.pdf) to ResNet.
- Conditioning a network on a side input is a scale and a shift per feature ([FiLM](https://mlanthology.org/distill/2018/dumoulin2018distill-featurewise)), the same move as AdaLN in diffusion transformers.
- Top k routing needs a load balancing term ([Shazeer and colleagues 2017](https://arxiv.org/pdf/1701.06538)).
- Structure follows symmetry ([Geometric Deep Learning](https://arxiv.org/abs/2104.13478)) and relations ([Battaglia and colleagues 2018](https://arxiv.org/abs/1806.01261)). Structure that matches the solving algorithm needs fewer examples ([Xu and colleagues 2020](https://arxiv.org/abs/1905.13211)).
- Diagnose before changing the model, training error against test error ([Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/), [ResNet](https://arxiv.org/abs/1512.03385)).
- Broad true assumptions last and narrow tricks lose to scale ([the bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)).
- Small synthetic tasks that isolate one capability are how architectures get tested ([Zoology](https://arxiv.org/abs/2312.04927)).

## Research on teaching

- Karpathy's advice on expertise is concrete projects taken one after another, depth first, learning what each needs, and teaching what you learn ([summary](https://www.goodreads.com/author_blog_posts/24921801-the-mental-equivalent-of-sweating?tab=book)). Deliberate practice asks for small tasks with clear goals and immediate feedback ([summary](https://td.org/insights/science-of-learning-101-what-kind-of-practice-makes-expert)). So the course pairs tiny exercises with a project at the end of each part.
- Building a thing from scratch, then using the library version, is how Dive into Deep Learning checks understanding ([section](https://D2L.AI/chapter_linear-regression/linear-regression-scratch.html)).
- Mayer's principles ([summary](https://cloud.wikis.utexas.edu/wiki/x/8QJqAw)): cut what is not needed, highlight the key point, split material into small steps, teach the names of things before using them.
- Plain language guidelines ([quick guide](https://www.stylemanual.gov.au/style-manual-resources/quick-guides/quick-guide-plain-language), [top ten](https://centerforplainlanguage.org/plain-language-top-ten-making-words-work/)): short sentences with one idea each, active voice, every term defined where it first appears.
- Concrete examples before the abstract idea ([concreteness fading](https://www.doi.org/10.1007/S10648-014-9249-3)), and contrasting a failing case with a working one before explaining ([A time for telling](https://www.wright.edu/sites/www.wright.edu/files/uploads/2017/Feb/event/Schwartz_Bransford_1998_TimeForTelling.pdf)).

## What failed with the owner in AIlings

- Lessons in README files, which the owner never opens.
- Exercises that were only a prediction to type in, which read as a quiz with no code to write.
- Text that assumed knowledge the owner had forgotten, such as why a Linear layer exists at all.
- A watch screen that scrolled instead of redrawing in place.
- A scope too narrow for the owner's goal, only architectures and only one framing of them.

## What failed with the owner in phase 1a

- Lessons that read as machine written and were hard to follow. They opened with definitions, put bold labels like "Why it exists." on paragraphs, chained "which is a ..." clauses, and gave every paragraph the same job in the same order.
- Lessons that held the answer. In all six exercises the solution line was nearly a copy of a line in the lesson, so the task was retyping.
- A result box that listed only the failing tests, so the learner could not find the function that already worked.
- After the fix for copied answers, lessons that left out tools the tasks needed, such as `.numel()`, and tasks that were puzzles with no purpose, such as a times table. The owner called them leetcode and asked for rustlings or Zero to Hero style, which led to building the makemore bigram model through the exercises.
- Story examples, such as a shop selling apples and pears, and vague comparisons.

## Research on writing that sounds human

- Signs of machine writing ([Wikipedia's guide](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), [Julia Evans on confusing explanations](https://jvns.ca/blog/confusing-explanations/)): opening on a definition, bold labels on paragraphs, clauses bolted onto the end of a sentence, grouping in threes, tidy upbeat closers, overexplaining, and swapping synonyms for one thing. Instruction tuned models use participle clauses and nominalizations far more than people do ([Reinhart et al. 2025](https://arxiv.org/html/2410.16107v2)). Expert readers spot predictable text with no person or surprise in it ([Russell et al. 2025](https://arxiv.org/abs/2501.15654)).
- How good teachers write. Karpathy shows a number moving before he names the derivative, and opens his training recipe on real failures ([micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0), [recipe](https://karpathy.github.io/2019/04/25/recipe/)). fast.ai plays the whole game first ([fastbook](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb)). The Rust Book compiles broken code on purpose and lets the error bring in the next idea ([chapter 2](https://doc.rust-lang.org/book/ch02-00-guessing-game-tutorial.html)).
- What the course takes from this. Open with a situation, show code and its output before the name, one new idea per paragraph, talk to "you", no bold labels, and give the why as a case the learner will meet.

## Research on tasks that make the learner think

- Worked examples help beginners and stop helping as skill grows, so later examples should leave more blank ([worked example effect](https://en.wikipedia.org/wiki/Worked-example_effect)). Trying a problem before instruction helps with concepts ([Sinha and Kapur 2021](https://doi.org/10.3102/00346543211019105)). Producing an answer sticks better than reading one, and retyping a shown answer feels fluent but teaches little (Bjork and Bjork 2011).
- A feature is only noticed when it varies while the rest stays still ([variation theory](https://kb.edu.hku.hk/approaches_variation_theory/)). So the lesson shows one case and the task changes one thing, another dimension, another shape or another operation.
- Rustlings states the goal in words and leaves the syntax to the book or a hint ([hints](https://github.com/rust-lang/rustlings/blob/main/rustlings-macros/info.toml)). Hints should climb from where to look, to the idea, to the answer, and learners who can click freely skip to the answer ([Cody et al. 2021](https://ar5iv.labs.arxiv.org/html/2102.05741)).
- Asking for a prediction before running code helps in general ([PRIMM](https://static.raspberrypi.org/files/curriculum/quickreads/11-Pedagogy_Summary_PRIMM_V4_2023.pdf)), but the owner rejected prediction exercises, so lessons do not ask for them.

## Research on the app's library

- Claude Code draws its screen with Ink, React for the terminal. What makes it pleasant is a fixed bottom bar with dim key hints, tool output folded to one line until opened, mouse wheel scrolling, and few colors ([fullscreen](https://code.claude.com/docs/en/fullscreen), [Ink](https://github.com/vadimdemedes/ink)).
- In Python, Textual has scrolling panes that take focus, the mouse wheel, collapsible sections, Markdown with highlighted code, a footer built from key bindings, thread workers for blocking work, and headless tests with SVG screenshots ([docs](https://textual.textualize.io/)). prompt_toolkit and urwid give layout parts with no Markdown or footer.
- Textual's costs. The mouse is captured, so copying needs the app's own drag selection or holding Fn in Terminal and Shift in VS Code. Copy goes through OSC 52, which macOS Terminal ignores, so the app also calls `pbcopy`. Textual has no file watcher, and its own reloader polls file times. The company behind it closed in 2025 and Textual continues as open source.
