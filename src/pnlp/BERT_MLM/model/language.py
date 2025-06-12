"""Protein Language Models"""

import torch
import torch.nn as nn
import lightning as L

from pnlp.BERT_MLM.model.bert import BERT

class ProteinMaskedLanguageModel(L.LightningModule):
    """Masked language model for protein sequences"""

    def __init__(self, hidden: int, vocab_size: int):
        """
        hidden: input size of the hidden linear layers
        vocab_size: vocabulary size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.linear(x)

# %%
class ProteinLM(L.LightningModule):
    """
    BERT protein language model
    """

    def __init__(self, bert: BERT, vocab_size: int):
        super().__init__()
        self.bert = bert
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x):
        x = self.bert(x)
        return self.mlm(x)
