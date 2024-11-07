import os
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
from collections import Counter, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# NLTK resources and path setting
nltk.data.path.append(os.path.expanduser('~/nltk_data/'))
nltk.download('punkt', download_dir=os.path.expanduser('~/nltk_data/'))
nltk.download('stopwords', download_dir=os.path.expanduser('~/nltk_data/'))
stop_words = set(stopwords.words('english'))

# Preprocess function for tokenizing, lowercasing, and removing stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [re.sub(r'\W+', '', token) for token in tokens if token not in stop_words and token.isalpha()]
    return tokens

# Function to apply preprocessing to a specified DataFrame
def prepare_dataframe(df, text_column='selftext'):
    df['tokens'] = df[text_column].apply(preprocess_text)
    return df

# Function to calculate word and word-pair frequencies
def calculate_frequencies(df, target_word, window_size=10, min_word_freq=10, min_pair_freq=4):
    word_freq = Counter()
    pair_freq = defaultdict(int)
    
    for tokens in df['tokens']:
        word_freq.update(tokens)
        for i, word in enumerate(tokens):
            if word == target_word:
                window = tokens[i+1:i+window_size+1]
                for pair in window:
                    if word != pair:
                        sorted_pair = tuple(sorted([word, pair]))
                        pair_freq[sorted_pair] += 1

    filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq >= min_word_freq}
    filtered_pair_freq = {pair: freq for pair, freq in pair_freq.items() if freq >= min_pair_freq}
    
    print(f"\nFiltered Word Frequency for '{target_word}':")
    if target_word in filtered_word_freq:
        print(f"{target_word}: {filtered_word_freq[target_word]}")
    
    return filtered_pair_freq

# Function to create and display a PMI graph for a specified target word
def plot_pmi_graph(filtered_pairs, target_word, subreddit_of_interest, threshold=2.0, max_related_words=6):
    pmi_scores = {pair: np.log2(freq + 1) for pair, freq in filtered_pairs.items()}
    
    filtered_pairs = {pair: pmi for pair, pmi in pmi_scores.items() if pmi > threshold and target_word in pair}
    top_related_pairs = dict(sorted(filtered_pairs.items(), key=lambda x: x[1], reverse=True)[:max_related_words])

    G = nx.Graph()
    for (word1, word2), pmi in top_related_pairs.items():
        G.add_edge(word1, word2, weight=pmi)

    pos = {target_word: [0.5, 0.5]}
    angle_step = 2 * np.pi / len(top_related_pairs)
    for j, ((word1, word2), pmi) in enumerate(top_related_pairs.items()):
        angle = j * angle_step
        distance = max(0.1, 1 / (pmi + 1))
        x = 0.5 + distance * np.cos(angle)
        y = 0.5 + distance * np.sin(angle)
        pos[word2 if word1 == target_word else word1] = [x, y]

    plt.figure(figsize=(7, 6))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, font_size=11)
    nx.draw_networkx_edges(G, pos)
    edge_labels = {(word1, word2): f"{pmi:.1f}" for (word1, word2), pmi in top_related_pairs.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.title(f"{subreddit_of_interest}")
    plt.show()