# HealthNER
## *Named Entity Recognition for Chinese Healthcare Applications*
Named Entity Recognition is a fundamental task in
information extraction, which locates and classifies defined named
entities in unstructured text. Chinese NER is more difficult than
English NER. Since there are no separators between Chinese
characters, incorrectly segmented entity boundaries will cause
error propagation in NER. In this study, named entity recognition
is constructed and applied in the Chinese medical domain, where
Chinese medical datasets are labeled in BIO format. The Chinese
HealthNER Corpus contains 33,896 sentences, of which 2531
sentences are divided into the validation set and 3204 sentences are
divided into the test set. This study uses PyTorch Embedding +
BiLSTM + CRF, RoBERTa + BiLSTM + CRF, BERT Classifier,
and BERT + BiLSTM + CRF for training and compares their
model performance. Finally, the BERT + BiLSTM + CRF achieves
the best prediction performance with a precision of 91.30%, recall
of 89.46%, and F1 score of 90.53%
