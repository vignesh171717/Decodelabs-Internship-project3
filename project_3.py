from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("      AI RECOMMENDATION SYSTEM USING TF-IDF")
print("=" * 60)

courses = [
    "Python programming for beginners",
    "Advanced Python programming",
    "Python for data analysis",

    "Artificial intelligence(AI) fundamentals",
    "Machine learning with Python for AI",
    "Deep learning and neural networks for AI",
    "Natural language processing for AI",

    "Data science and analytics",
    "Data visualization with Python",
    "Big data technologies",

    "Web development using HTML CSS JavaScript",
    "Frontend development with React",
    "Backend development with Django",

    "Cyber security fundamentals",
    "Ethical hacking and penetration testing",
    "Network security basics",

    "Graphic design and creative tools",
    "UI UX design principles",
    "Adobe Photoshop essentials"
]

topics = [
    "Python",
    "AI",
    "Data Science",
    "Web Development",
    "Cyber Security",
    "Graphic Design"
]

print("\nAvailable Topics:\n")

for topic in topics:
    print("•", topic)

user_interest = input("\nEnter your interest: ").lower()

courses_lower = [course.lower() for course in courses]

documents = [user_interest] + courses_lower

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

user_vector = tfidf_matrix[0]
course_vectors = tfidf_matrix[1:]

similarity_scores = cosine_similarity(
    user_vector,
    course_vectors
)

results = list(zip(courses, similarity_scores[0]))
results.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 60)
print("RECOMMENDED COURSES")
print("=" * 60)

found = False

for course, score in results:
    if score > 0:
        print("•", course)
        found = True

if not found:
    print("\nNo matching recommendations found.")
    print("\nTry one of these topics:")

    for topic in topics:
        print("•", topic)

print("\nThank you for using the AI Recommendation System!")