
# Project 1: Rule-Based AI Chatbot


responses = {
    "hello":       "Hello! I'm KashBot. How can I help you today?",
    "hi":          "Hi there! Ask me anything.",
    "hey":         "Hey! Great to see you. What's on your mind?",
    "who are you":  "I'm KashBot, a rule-based AI built by Kashvi!",
    "what are you": "I'm a rule-based chatbot — no ML, pure logic!",
    "help":         "I can answer: hello, time, joke, ai, weather, bye.",
    "what can you do": "I respond to keywords using a Python dictionary!",
    "joke":         "Why did the AI fail the test? It had too many bugs!",
    "tell me a joke": "I told my GPU a joke... it said: 'cuda not compute.'",
    "what is ai":   "AI is a machine simulating human intelligence!",
    "ai":           "AI stands for Artificial Intelligence — you're talking to one!",
    "machine learning": "ML is AI that learns from data — this bot is rule-based, not ML!",
    "python":       "Python is the best language for AI. Great choice!",
    "decodelabs":   "DecodeLabs is where you're doing your AI internship!",
    "time":         "I don't have a clock, but your device does!",
    "weather":      "I can't check live weather, but it looks great outside!",
    "bye":          "Goodbye! Keep building awesome things!",
    "goodbye":      "See you later! Come back if you have questions.",
    "thanks":       "You're welcome! Happy to help.",
    "thank you":    "Anytime! That's what I'm here for.",
}

FALLBACK = "I don't understand that yet. Try: hello, help, joke, or bye."

print("KashBot v1.0 — Rule-Based AI Chatbot")
print("Type 'help' to see what I can do.")
print("Type 'exit' to quit.")


while True:
    raw_input = input("You: ")
    clean_input = raw_input.lower().strip()
    if clean_input == "exit":
        print("Bot: Shutting down. Goodbye!")
        break
    if not clean_input:
        continue
    reply = responses.get(clean_input, FALLBACK)
    print(f"Bot: {reply}")