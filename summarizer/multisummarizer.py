import glob
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk import sent_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import networkx as nx

#Read files
path = input('Please Enter Directory Path: ')
print ('Reading files...')
path = path + "/*.txt"
files = glob.glob(path)
docs = []
for name in files: 
    with open(name) as f: 
        docs.append(f.read())


#add title as headline
fname = [fname[12:] for fname in files]
fname = [re.sub(".txt",". ",n) for n in fname]
docs = list(map(str.__add__,fname,docs))

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

#Defining pre-process function
def pre_process(doc):
    tokens = [word for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]
    tokens= [w for w in tokens if w not in stopwords]
    tokens = [w.lower() for w in tokens if w.isalpha()]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def summary_sents(sub_topic, sents):
    sents = sent_tokenize(' '.join(sents))
    sents.append(sub_topic)
    
    sent_vectorizer = TfidfVectorizer(decode_error='replace',min_df=1, stop_words='english',
                                 use_idf=True, tokenizer=pre_process, ngram_range=(1,3))

    sent_tfidf_matrix = sent_vectorizer.fit_transform(sents)
    #subtopic_tfidf = sent_tfidf_matrix.transform([sub_topic])
    sub_topic_similarity = cosine_similarity(sent_tfidf_matrix)
    top10_sents = sub_topic_similarity[-1][:-1].argsort()[:-11:-1]
    final_sents=[]
    for i in top10_sents:
        final_sents.append(sents[i])
    return final_sents

def summarize(text):
    
    sentences_token = sent_tokenize(text)
    
    #Feature Extraction
    vectorizer = CountVectorizer(min_df=1,decode_error='replace')
    sent_bow = vectorizer.fit_transform(sentences_token)
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
    sent_tfidf = transformer.fit_transform(sent_bow)
    
    similarity_graph = sent_tfidf * sent_tfidf.T

    
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    text_rank_graph = sorted(((scores[i],s) for i,s in enumerate(sentences_token)), reverse=True)
    print(scores)
    number_of_sents = int(0.4*len(text_rank_graph))
    del text_rank_graph[number_of_sents:]
    summary = ' '.join(word for _,word in text_rank_graph)
    
    return summary

print ('Clustering Documents...')

#tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.8,max_features=200000,decode_error='replace',
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=pre_process, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
terms = tfidf_vectorizer.get_feature_names()

#Clustering
corpus_similarity = cosine_similarity(tfidf_matrix)#Similarity
km = KMeans(n_clusters=3)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# print(classification_report(tfidf_matrix,km.labels_))

print ('Modelling Topics...')

#topic extraction
lda = LatentDirichletAllocation(n_components=3,max_iter=200,learning_method='online',
                                learning_offset=50.,random_state=42)
lda.fit(tfidf_matrix)

topics=[]
for topic_idx, topic in enumerate(lda.components_):
    topics.append(" ".join([terms[i]
                        for i in topic.argsort()[:-30 - 1:-1]]))



#finding similar topic
query = input('Please Enter Topic: ')
topics.append(query)
tfidf_topics_matrix = tfidf_vectorizer.fit_transform(topics)
topic_similarity = cosine_similarity(tfidf_topics_matrix)
topics=topics[:-1]

#index of most similar cluster
similar_clust = np.argmax(topic_similarity[3][:3])
article_indices = [i for i, x in enumerate(clusters) if x == similar_clust]

print ('Building Extractive Summary...')

#extractive summary
ext_summary = []
for i in article_indices:
    ext_summary.append(summarize(docs[i] ))

# print(len(ext_summary))
# exit()

#subtopic relevant summary
sub_topic=[]
sub_topic.append(input('Please Enter Sub Topic 01: '))
sub_topic.append(input('Please Enter Sub Topic 02: '))
print ('Making Bullet Points...')

summary_bullets = summary_sents(sub_topic[0], ext_summary)
print ("\n"+sub_topic[0]+"\n")
for b in summary_bullets:
    print ('* '+b)

summary_bullets = summary_sents(sub_topic[1], ext_summary)
print ("\n"+sub_topic[1]+"\n")
for b in summary_bullets:
    print ('* '+b)