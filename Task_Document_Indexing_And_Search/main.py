from collections import Counter
import operator
import math


def tokenize(doc):
    words = [word.replace(',', '').lower() for word in doc.split()]
    return words


def build_terms(corpus):
    terms = {}
    current_index = 0
    for doc in corpus:
        for word in tokenize(doc):
            if word not in terms:
                terms[word] = current_index
                current_index += 1
    return terms


def tf(document, terms):
    words = tokenize(document)
    total_words = len(words)
    doc_counter = Counter(words)
    for word in doc_counter:
        doc_counter[word] /= total_words
    tfs = [0 for _ in range(len(terms))]
    for term, index in terms.items():
        tfs[index] = doc_counter[term]
    return tfs


def _count_docs_with_word(word, docs):
    counter = 1
    for doc in docs:
        if word in doc:
            counter += 1
    return counter


def idf(documents, terms):
    idfs = [0 for _ in range(len(terms))]
    total_docs = len(documents)
    for word, index in terms.items():
        docs_with_word = _count_docs_with_word(word, documents)
        idf = 1 + math.log10(total_docs / docs_with_word)
        idfs[index] = idf
    return idfs


def _merge_td_idf(tf, idf, terms):
    return [tf[i] * idf[i] for i in range(len(terms))]


def build_tfidf(corpus, document, terms):
    doc_tf = tf(document, terms)
    doc_idf = idf(corpus, terms)
    return _merge_td_idf(doc_tf, doc_idf, terms)


def cosine_similarity(vec1, vec2):
    def dot_product2(v1, v2):
        return sum(map(operator.mul, v1, v2))

    def vector_cos5(v1, v2):
        prod = dot_product2(v1, v2)
        len1 = math.sqrt(dot_product2(v1, v1))
        len2 = math.sqrt(dot_product2(v2, v2))
        return prod / (len1 * len2)

    return vector_cos5(vec1, vec2)


doc0 = "В мире нет ничего, что можно знать наверняка. Это единственное, что я знаю наверняка."
doc1 = "Если не спросить — никогда не узнаешь, если знаешь — нужно лишь спросить."
doc2 = "Мудрость — это применение знаний."
doc3 = "Верные слова не изящны. Красивые слова не заслуживают доверия. "

check_str1 = "верные слова"
check_str2 = "мудрость"
check_str3 = "это никогда не понять"

tf_idf_total = []
corpus = (doc0, doc1, doc2, doc3)
terms = build_terms(corpus)

for document in corpus:
    tf_idf_total.append(build_tfidf(corpus, document, terms))

queries = (check_str1, check_str2, check_str3)
for query in queries:
    print("QUERY:", query)
    query_tfidf = build_tfidf(corpus, query, terms)
    for index, document in enumerate(tf_idf_total):
        print("Similarity with DOC", index, "=", cosine_similarity(query_tfidf, document))
