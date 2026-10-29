"""Every chapter and exercise, in the order the learner meets them.

Each chapter has its own module. A quiz lives in the module of the chapter it
closes, and goes in the quizzes folder like every quiz."""

from __future__ import annotations

from ..course import QUIZZES, Chapter, Exercise
from . import autograd, chain_rule, derivatives, matmul, tensor_backprop, tensors

CHAPTERS: list[Chapter] = [
    tensors.TENSORS,
    matmul.MATMUL,
    QUIZZES,
    derivatives.DERIVATIVES,
    chain_rule.CHAIN_RULE,
    autograd.AUTOGRAD,
    tensor_backprop.TENSOR_BACKPROP,
]

EXERCISES: list[Exercise] = [
    tensors.TENSORS1,
    tensors.INDEXING1,
    tensors.MATH1,
    tensors.BROADCASTING1,
    tensors.REDUCTIONS1,
    matmul.MATMUL1,
    matmul.QUIZ1,
    derivatives.DERIVATIVES1,
    derivatives.DERIVATIVES2,
    derivatives.DERIVATIVES3,
    chain_rule.CHAIN1,
    chain_rule.CHAIN2,
    chain_rule.CHAIN3,
    autograd.AUTOGRAD1,
    autograd.AUTOGRAD2,
    autograd.AUTOGRAD3,
    autograd.AUTOGRAD4,
    tensor_backprop.BACKPROP1,
    tensor_backprop.BACKPROP2,
    tensor_backprop.BACKPROP3,
    tensor_backprop.BACKPROP4,
    tensor_backprop.QUIZ2,
]
