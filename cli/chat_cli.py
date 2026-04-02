from app.rag_pipeline import ask_question

print("\n📚 RAG Knowledge Assistant")
print("Type 'exit' to quit\n")

while True:
    
    query = input("Ask a question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    answer, sources = ask_question(query)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:")
    for s in sources:
        print("-", s)

    print("\n---------------------------\n")