**分词器的训练和合并**
- 中文医疗领域分词器的训练
- 中文医疗领域分词器和Llama2原始分词器的合并

**分词器的训练**
- 使用脚本train_medical_tokenizer.py训练
- 将语料切割成句子，分隔符为中英文的问好、叹号、句号、分号，并保留分隔符
- 以句子为单位，使用sentencepiece进行训练，分词算法为bpe，词表大小设为30000

**分词器的合并**
- 使用脚本merge_tokenizer.py进行分词器的合并
- 合并方法可参看[Google sentencepiece官方合并教程](https://github.com/google/sentencepiece/blob/master/python/add_new_vocab.ipynb)
- 合并逻辑：由于bpe算法不是基于概率进行切词的，而是基于token长度从大到小进行切词，因此可以直接合并两者的token词表

**分词器合并逻辑**
```python
for p in chinese_medical_spm.pieces:
    piece = p.piece
    if piece not in llama_tokens:
        new_p = sentencepiece_model_pb2.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
```

**分词器合并结果**
|分词器|词表大小|
|:--|:--|
|Llama2|32000|
|中文医疗|30000|
|合并后分词器|60912|