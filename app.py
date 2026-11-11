from semantic_cache import SemanticCache

llm_calls = 0


def ask_llm(question):
    # Stand-in for your real model call (OpenAI, Anthropic, Gemini, a local model).
    global llm_calls
    llm_calls += 1
    return f"LLM answer #{llm_calls}"


cache = SemanticCache(threshold=0.85)


def answer(question):
    vector = cache.embed(question)
    cached, score = cache.lookup(vector)
    if cached is not None:
        print(f"HIT   {score:.2f}  {question}  ->  {cached}")
        return cached
    reply = ask_llm(question)
    cache.store(question, vector, reply)
    print(f"MISS  {score:.2f}  {question}  ->  {reply}")
    return reply


questions = [
    "How do I reset my password?",
    "I forgot my password. How do I reset it?",
    "Help, I can't reset my password.",
    "how can i change my password",
    "How do I change my email address?",
    "Can I update the email on my account?",
    "My password reset email never arrived",
]

for question in questions:
    answer(question)

print(f"{len(questions)} questions, {llm_calls} LLM calls")
