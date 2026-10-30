---
name: DEC-019-three-image-projects-give-way-to-exoplanets-dna-and-a-citation-graph
kind: decision
description: on the owner's word, Sections 2, 5 and 8 drop their MNIST, CIFAR and SimCLR projects for an exoplanet hunter, a DNA motif finder and a paper topic map, and Section 9 keeps its digits
updated: 2026-10-30
links: []
date: 2026-10-30
area: course
status: standing
---

## Decision

Section 2's project is now 14_exoplanets, which tells real Kepler planets from false alarms with softmax regression and an MLP. Section 5's is 37_dna, a 1D conv net that finds the motifs in real DNA sequences. Section 8's is 64_paper_topics, which learns embeddings of the Cora citation graph from random walks with no labels and probes them for topics with 140 labels. They replace 14_mnist, 37_cifar and 64_probe. On 2026-10-30 the owner said "Offer replacement for the image ones, I really don't like image models" and picked these. Section 9's digit diffusion stays at the owner's choice of "Keep digits".

## Alternatives

- Section 2: surname to language and a spam filter were offered first and got "Meh, generate more." The author detective and the code language guesser were offered second, and the exoplanet hunter won.
- Section 5: hearing digits from raw audio and a Connect Four judge were offered. The owner picked DNA, and later ruled out audio.
- Section 8: SimCSE on Shakespeare and a masked pretrained probe were turned down with "Generate something else". Movie and molecule embeddings were offered second.
- Section 9: a diffusion path planner and a folk tune writer were offered, and the owner kept the digits.

## Consequences

The MNIST and CIFAR datasets stay, since chapters use them, and the helpers other modules import from mnist.py and cifar.py stay with them. Every lesson that named mnist1 to mnist3, cifar1 to cifar3 or probe1 to probe3 is repointed. The reference checkpoints cifar2.pt and probe1.pt are no longer used. 96_papers still reproduces an image paper, the squeeze and excitation block, which the owner may want swapped.
