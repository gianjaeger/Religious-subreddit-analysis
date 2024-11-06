import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nltk
from textblob import TextBlob

def plot_subreddit_term_space(vectors, term1, term2, title=None):
    """
    Plot subreddit vectors in a 2D term space. Original code adapted to work with more than 3 subreddits.
    
    Parameters:
    - vectors: dict with subreddit names as keys and np.arrays as values
    - term1: string name of first term (x-axis)
    - term2: string name of second term (y-axis)
    - title: optional custom title
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Plot vectors from origin
    colors = cm.get_cmap('tab10', len(vectors))

    # for (name, vec), color in zip(vectors.items(), colors):
    #     plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
    #               color=color, label=name, width=0.008)

    for idx, (name, vec) in enumerate(vectors.items()):
        plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
                   color=colors(idx), label=name, width=0.008)
    
    # Style the plot
    plt.grid(True, linestyle='--', alpha=0.7)

    # Fix: stack all vectors and find max value
    all_values = np.concatenate([v for v in vectors.values()])
    max_val = np.max(all_values) * 1.2
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)

    ax.set_aspect('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Labels
    plt.xlabel(f"'{term1}' TF-IDF score")
    plt.ylabel(f"'{term2}' TF-IDF score")
    plt.title(title or f"Subreddit Vectors in {term1}-{term2} Space")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def report_distances(vectors):
    """
    Report the distances between subreddit vectors.
    
    Parameters:
    - vectors: dict with subreddit names as keys and np.arrays as values
    """
    for name1, vec1 in vectors.items():
        for name2, vec2 in vectors.items():
            if name1 < name2:
                dist = np.linalg.norm(vec1 - vec2)
                print(f"Distance between {name1} and {name2}: {dist:.2f}")
                
    # Print angles between vectors
    print("\nAngles between subreddit vectors:")
    for name1, vec1 in vectors.items():
        for name2, vec2 in vectors.items():
            if name1 < name2:  # avoid duplicate comparisons
                cos_sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                angle = np.degrees(np.arccos(cos_sim))
                print(f"{name1} vs {name2}: {angle:.1f}°")

def create_tf_idf_vectors(results, subs_of_interest, word1, word2):
    """
    Creates vectors of TF-IDF scores for two words across multiple subreddits.

    Parameters:
    - results: Dictionary containing TF-IDF scores by subreddit, with scores for each word.
    - subs_of_interest: List of subreddit names we want to create vectors for.
    - word1: First word to use for TF-IDF coordinates.
    - word2: Second word to use for TF-IDF coordinates.

    Returns:
    - A dictionary where keys are subreddit names and values are numpy arrays 
      containing the TF-IDF scores of word1 and word2 for each subreddit.
    """
    # Initialize a dictionary to hold the vectors for each subreddit
    tf_idf_vectors = {}
    
    for subreddit in subs_of_interest:
        # Extract TF-IDF scores for both words in the current subreddit
        try:
            tf_idf_word1 = results[subreddit]['tf_idf_scores'].loc[word1, 'score']
            tf_idf_word2 = results[subreddit]['tf_idf_scores'].loc[word2, 'score']
            
            # Create a numpy array with the TF-IDF scores of the two words
            tf_idf_vectors[subreddit] = np.array([tf_idf_word1, tf_idf_word2])
        
        except KeyError as e:
            print(f"Warning: Could not find TF-IDF score for {e} in subreddit {subreddit}.")
            tf_idf_vectors[subreddit] = np.array([np.nan, np.nan])
    
    return tf_idf_vectors

def process_subreddit_subjectivity(results, subreddit_name):
    """
    Extracts the posts DataFrame for a given subreddit, filters out empty 'selftext',
    and calculates the subjectivity of each post.

    Parameters:
    - results (dict): Dictionary containing subreddit dataframes.
    - subreddit_name (str): The name of the subreddit to process.

    Returns:
    - DataFrame: The processed DataFrame with a new 'subjectivity' column.
    """
    df = results[subreddit_name]['posts_df']
    df = df[df['selftext'].str.strip().astype(bool)].copy()
    df['subjectivity'] = df['selftext'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df