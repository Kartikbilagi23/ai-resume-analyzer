"""
Notes that would help for quick revision.
Codes are commented;use when required ;)
"""
"""----------------------1.Model testing and Intro----------------------"""
# from transformers import pipeline

# classifier = pipeline("sentiment-analysis")

# # Test sentence
# result = classifier("I am learning Hugging Face from scratch")

# print(result)
# from transformers import pipeline

# generator=pipeline("text-generation",model='gpt2')

# result=generator(
#     "Machine learning is",
#     max_length=40,
#     num_return_sequences=1
# )
# print(result[0]["generated_text"])
# from transformers import pipeline

# summarizer = pipeline("summarization")

# text = """
# Machine learning allows computers to learn from data without being explicitly programmed.
# It is a core part of artificial intelligence and is used in many real-world applications.
# """

# result = summarizer(text, max_length=50, min_length=25)....1.22gb ...heavy model
# print(result)


# from transformers import pipeline

# qa=pipeline("question-answering")
# context="""
# Hugging Face provides tools and models for machine learning.
# It is widely used for natural language processing tasks.
# """

# question="What is Hugging Face used for?"
# result = qa({
#     "question": question,
#     "context": context
# })
# print(result)
# from transformers import pipeline

# summarizer=pipeline(
#     "summarization",
#     model="sshleifer/distilbart-cnn-12-6"
# )
# text = """
# Machine learning allows computers to learn from data without being explicitly programmed.
# It is a core part of artificial intelligence and is used in many real-world applications.
# """
# print(summarizer(text,max_length=40,min_length=20))  ....1.22gb..heavy model
# from transformers import pipeline

# summarizer = pipeline(
#     "summarization",
#     model="t5-small"
# )

# text = """
# Machine learning allows computers to learn from data without being explicitly programmed.
# It is a core part of artificial intelligence and is used in many real-world applications.
# """

# print(summarizer(text, max_length=40, min_length=20))....light-weight..240mb

"""
----------------2.Tokeniser--------------------------------------
"""
# from transformers import AutoTokenizer
# tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')

# text="I am learning hugging face"

# tokens=tokenizer(text)
# print(tokenizer.tokenize(text))

"""
Models have fixed size input.

If text is too long:
❌ Error
❌ Truncation
❌ Wrong output
"""
# e.g.
# long_text = "Hugging Face is amazing. " * 50

# tokens = tokenizer(
#     long_text,
#     truncation=True,
#     max_length=20
# )

# print(len(tokens["input_ids"]))

"""
print(tokens)..o/p->{'input_ids': [101, 1045, 2572, 4083, 17662, 2227, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
print(tokenizer.tokenize(text))..o/p->['i', 'am', 'learning', 'hugging', 'face']
print(len(tokens["input_ids"]))...o/p->20
"""


"""
e.g  of (1)-----
op-1->Machine learning is a process that involves a number of discrete tasks, but that involves a large number of processors. By using a wide 
range of processors, we can make deep learning the fastest, easiest, and most efficient way to learn algorithms. We found that in the case of deep learning, there is a considerable amount of complexity involved, and that learning algorithms that are used by a large number of processor architectures are also capable of learning algorithms that are much more complex than these.

We also found that the total number of CPU cores and memory chips used for deep learning is often much higher than the number of CPU cores and memory chips used by other deep learning platforms. In fact, the number of CPUs and memory chips used by deep learning platforms 
is often far more than the total number of CPU cores and memory chips used by other deep learning platforms.

The main finding we found in this study was that deep learning methods have a particularly high number of CPUs. The number of CPUs and memory chips used for deep learning has been increasing over the years, and while in fact that is true in some cases, it is not true in others. The number of CPUs and memory chips used in deep learning for many years has been decreasing, at the same time these CPUs and memory chips are being used

op2->{'score': 0.5089405179023743, 'start': 84, 'end': 117, 'answer': 'natural language processing tasks'}
op3-> taking 1.22 gb so i didnt attempted
"""

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# model = AutoModelForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased-finetuned-sst-2-english"
# )


# text = "I really enjoy learning Hugging Face"

# inputs = tokenizer(text, return_tensors="pt")
# print(inputs)
"""
o/p->{'input_ids': tensor([[  101,  1045,  2428,  5959,  4083, 17662,  2227,   102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 
1, 1, 1]])}
"""
# with torch.no_grad():
#     outputs=model(**inputs)

# print(outputs)
"""
o/p->SequenceClassifierOutput(loss=None, logits=tensor([[-3.8382,  
4.1143]]), hidden_states=None, attentions=None)
"""
# logits = outputs.logits
# pred = torch.argmax(logits, dim=1)

# labels = ["NEGATIVE", "POSITIVE"]
# print(labels[pred.item()])
"""
o/p->POSITIVE
"""
"""
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Transformer
 ↓
Logits
 ↓
Argmax / Softmax
 ↓
Prediction

🧠 What We ACTUALLY Did in above code
input_ids = [101, 1045, 2428, 5959, 4083, 17662, 2227, 102]
101->CLS summary token
102-> SEP end
AutoModelForSequenceClassification-> a model that has transformer encoder(distilbert) and a classification head

outputs = model(**inputs)
This means:

“Given these token IDs, what does the model think?”

No weights were changed.
No training happened.
Only inference.
...............WHAT ARE LOGITS?.............
Logits are raw, unnormalized model scores
You got:

logits = [[-3.8382, 4.1143]]
They are:
NOT probabilities
NOT percentages
Because this model has 2 classes:
labels = ["NEGATIVE", "POSITIVE"]
logits[0] = score for NEGATIVE
logits[1] = score for POSITIVE
NEGATIVE → -3.83
POSITIVE →  4.11
👉 POSITIVE is much larger → model is confident 👍
...............
🤔 Why not directly output POSITIVE?
Because:
During training → loss functions (like CrossEntropyLoss) need logits

"""
"""
.................3.Embeddings................
"""
"""
Embeddings are numerical representations of meaning
"I love ML"
"I enjoy machine learning"
Different words ❌
Same meaning ✅

➡️ Their embeddings will be very close in vector space.

❌ NOT like token IDs
Token IDs:
"love" → 2293
Just dictionary index
NO meaning

Embeddings:
"love" → [0.21, -0.33, 0.89, ...]
Dense vectors
Capture semantics

"""
# from transformers import AutoTokenizer,AutoModel
# import torch

# tokenizer=AutoTokenizer.from_pretrained(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )
# model=AutoModel.from_pretrained(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )

# def get_embedding(sentence):
#     inputs=tokenizer(
#         sentence,
#         return_tensors="pt",
#         truncation=True,
#         padding=True
#     )
#     with torch.no_grad():
#         outputs=model(**inputs)

#     embedd=outputs.last_hidden_state.mean(dim=1)
#     return embedd

# s1 = "I love machine learning"
# s2 = "I enjoy studying AI"
# s3 = "The weather is very hot today"

# e1 = get_embedding(s1)
# e2 = get_embedding(s2)
# e3 = get_embedding(s3)
# from torch.nn.functional import cosine_similarity

# sim_12 = cosine_similarity(e1, e2)
# sim_13 = cosine_similarity(e1, e3)

# print("Similarity s1 & s2:", sim_12.item())
# print("Similarity s1 & s3:", sim_13.item())
"""
s1 vs s2----------------
“I love machine learning”
“I enjoy studying AI”
Different words ❌
Same intent ✅
Similarity ~0.54 → Correct
s1 vs s3-----------------
“The weather is very hot today”
Totally different topic ❌
Similarity ~0.02 → Correct
👉 This is why embeddings beat keyword search.
"""
# documents = [
#     "Machine learning is used for data analysis",
#     "Deep learning uses neural networks",
#     "Hugging Face provides pretrained models",
#     "The weather is hot today",
#     "AI is transforming healthcare"
# ]

# doc_emb=[]
# for doc in documents:
#     doc_emb.append(get_embedding(doc))

# query="How are AI models trained?"
# qu_emb=get_embedding(query)

# from torch.nn.functional import cosine_similarity
# scores=[]
# for emb in doc_emb:
#     score=cosine_similarity(qu_emb,emb)
#     scores.append(score.item())

# best_match_index=scores.index(max(scores))
# print("Best match:",documents[best_match_index])
# print("Score:",scores[best_match_index])

"""
Best match → "Deep learning uses neural networks"
Score → 0.43
“How are AI models trained?”

Closest meaning:
Training AI → learning patterns
Deep learning → neural networks
Training happens through neural networks
So the model correctly picked:
✅ Deep learning uses neural networks
Keyword search would FAIL here.
Embedding search succeeds.
"""
# best_doc = documents[best_match_index]
# prompt = f"""
# Context:
# {best_doc}

# Question:
# {query}

# Answer:
# """
# print(prompt)

# from transformers import pipeline

# generator = pipeline(
#     "text-generation",
#     model="distilgpt2"
# )

# output = generator(
#     prompt,
#     max_length=80,
#     do_sample=True,
#     temperature=0.7
# )

# print(output[0]["generated_text"])


