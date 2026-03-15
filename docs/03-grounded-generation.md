# Grounded Generation

In the previous section we built a **basic Retrieval Augmented Generation (RAG) pipeline**.

The system now retrieves relevant document chunks before generating an answer.

However, even with retrieval, the model can still produce **incorrect or unsupported statements**.

Why?

Because the model is still fundamentally a **text generator**, not a fact verifier.

To improve reliability we introduce a key concept:

**Grounded Generation**

---

# What is Grounded Generation?

Grounded generation means that the model should generate answers **strictly based on the retrieved context**.

Instead of answering from its internal knowledge, the model must rely on **external evidence provided in the prompt**.

The goal is simple:

> The answer should be supported by the retrieved documents.

---

# The Problem Without Grounding

Consider the following situation.

The system retrieves this context:
```
Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning.
```

The user asks:
```
When was Tesla founded?
```

The correct answer should be:
```
Tesla was founded in 2003.
```

However, an LLM might generate something like:
```
Tesla was founded in 2003 by Elon Musk.
```

The model combined real information with **incorrect assumptions**.

Even though RAG retrieved the correct document, the model still hallucinated.

This is why we need stronger grounding.

---

# Enforcing Grounded Responses

To enforce grounding, we modify the prompt so that the model can **only answer using the provided context**.

Example prompt:

Answer the question using ONLY the information in the provided context.
```
If the answer is not present in the context, respond with:
"Not found in documents."

Context:
{context}

Question:
{query}
```


This instruction tells the model:

- Do not use external knowledge
- Do not guess
- Use only the provided text

---

# Implementing Grounded Generation

In our pipeline, retrieved documents are combined into a single context string.

Example:

```python
docs = retrieve(index, query)

context = "\n".join([doc.page_content for doc in docs])

answer = generate_answer(query, context)
```
The context is injected into the prompt before generation.

The model now reasons over retrieved evidence instead of relying purely on its internal training data.

# Benefits of Grounded Generation

Grounded generation improves system reliability in several ways.

1. Reduced Hallucination

The model is constrained to information present in the retrieved documents.

2. Transparent Reasoning

Answers can be traced back to specific pieces of context.

3. Controlled Knowledge Scope

The model cannot invent information outside the provided documents.

This is particularly useful for:

- enterprise knowledge assistants

- internal documentation search

- legal and medical systems

# Limitations of Grounded Prompts

Even with strict prompts, grounded generation is not perfect.

The model may still:

- misinterpret context

- combine unrelated chunks

- produce unsupported conclusions

For example:
```
Context:
Tesla was founded in 2003.
Tesla produces electric vehicles.

Question:
Who founded Tesla?
```
If the retrieved chunk does not contain founder information, the model may still guess.

To handle these cases we need a verification layer.

# Next Step

Grounded prompts improve reliability, but they do not guarantee correctness.

In the next section we introduce verification guardrails, where a second model pass checks whether the generated answer is actually supported by the retrieved context.

This step allows the system to detect hallucinations before returning a response to the user.