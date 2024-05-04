import pandas as pd
import re
import info
from gensim.models import Word2Vec
from tqdm import tqdm
from collections import Counter

class Process:
    def __init__(self, df) -> None:
        self.df = df
        
        self.pb = tqdm(total=17)
        
        self.article_vocab_size = None
        self.highlights_vocab_size = None
        
        # Set the files to have all lower case
        self.setCase()
        
        # Replace contractions in DF
        self.replace_contractions()
        
        self.removeEmailIDs()
        self.removeURLs()
        
        self.removePossessives()
        self.removePunctuation()
        
        # self.removeStopWords()
        
        self.removeWhiteSpace()
        
        self.wordToVec()
        
    def getArticleVocabSize(self):
        if not self.article_vocab_size:
            word_counts = Counter(word for text in self.df['article'] for word in text.split())
            self.article_vocab_size = len(word_counts) + 1
        return self.article_vocab_size
    
    def getHighlightsVocabSize(self):
        if not self.highlights_vocab_size:
            word_counts = Counter(word for text in self.df['highlights'] for word in text.split())
            self.highlights_vocab_size = len(word_counts) + 1
        return self.highlights_vocab_size
            
        
    # Set the files to have all lower case
    # Turns Hi -> hi
    def setCase(self):
        self.pb.set_description(f'setCase - article')
        self.df['article'] = self.df['article'].str.lower()  # Using str accessor to apply lower() function
        self.pb.update(1)
        self.pb.set_description(f'setCase - highlights')
        self.df['highlights'] = self.df['highlights'].str.lower()  # Using str accessor to apply lower() function
        self.pb.update(1)

    
    # Replace contractions in DF
    # Turns "You're right." -> you are right
    def replace_contractions(self):
        contractions_dict = info.contractions_dict
        contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
        for column in ['article', 'highlights']:
            self.pb.set_description(f'replace contractions - {column}')
            self.df[column] = self.df[column].apply(lambda x: contractions_re.sub(lambda match: contractions_dict[match.group(0)], x))
            self.pb.update(1)
        
    # Removes all of the emails in the DF
    # Turns car@gmail.com -> null
    def removeEmailIDs(self):
        for column in ['article', 'highlights']:
            self.pb.set_description(f'remove EmailIDs - {column}')
            self.df[column] = self.df[column].apply(lambda x: re.sub('\S+@\S+','', x))
            self.pb.update(1)
        
    # Removes all of the URLs in the DF
    # Turns https://web.com -> null
    def removeURLs(self):
        for column in ['article', 'highlights']:
            self.pb.set_description(f'remove URLs - {column}')
            self.df[column] = self.df[column].apply(lambda x: re.sub("((http\://|https\://|ftp\://)|(www.))+(([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(/[a-zA-Z0-9%:/-_\?\.'~]*)?",'', x))
            self.pb.update(1)
        
    # Removes all of the URLs in the DF
    # Turns "She ran" -> She ran
    def removePossessives(self):
        for column in ['article', 'highlights']:
            self.pb.set_description(f'remove Possessives - {column}')
            self.df[column] = self.df[column].apply(lambda x: x.replace("'s", ''))
            self.df[column] = self.df[column].apply(lambda x: x.replace('’s', ''))
            self.df[column] = self.df[column].apply(lambda x: x.replace("\'s", ''))
            self.df[column] = self.df[column].apply(lambda x: x.replace("\’s", ''))
            self.pb.update(1)
        
    # Removing the Trailing and leading whitespace and double spaces
    # Turns "  Hi  " -> "Hi"
    def removeWhiteSpace(self):
        for column in ['article', 'highlights']:
            self.pb.set_description(f'remove White Space - {column}')
            self.df[column] = self.df[column].apply(lambda x: re.sub(' +', ' ',x))
            self.pb.update(1)
        
    # Removes the punctuation
    def removePunctuation(self):
        for column in ['article', 'highlights']:
            self.pb.set_description(f'remove punctuation - {column}')
            self.df[column] = self.df[column].apply(lambda x: ''.join(word for word in x if word not in info.punctuation))
            self.pb.update(1)
            
    # # Removing the Stop words
    # def removeStopWords(self):
    #     for column in ['article', 'highlights']:
    #         self.df[column] = self.df[column].apply(lambda x: ' '.join(word for word in x.split() if word not in info.stop_words))
        

    # Replace words with vectors in DF
    # Turns "run" -> [0.4234, 0.324 ... ]
    def wordToVec(self):
        # Tokenize the text
        tokenized_article = self.df['article'].apply(lambda x: x.split())
        tokenized_highlights = self.df['highlights'].apply(lambda x: x.split())
        
        # Train Word2Vec model
        # Ensure that tokenized_article and tokenized_highlights are lists of lists of words
        self.pb.set_description(f'training Word2Vec')
        self.modelWord2Vec = Word2Vec(tokenized_article.tolist() + tokenized_highlights.tolist(), vector_size=100, window=5, min_count=1, sg=0)
        self.pb.update(1)

        # Convert words to vectors
        self.pb.set_description(f'Word2Vec - article')
        self.df['article_vectors'] = tokenized_article.apply(lambda x: [self.modelWord2Vec.wv[word] for word in x if word in self.modelWord2Vec.wv])
        self.df['article_vectors'] = self.df['article_vectors'].apply(lambda x: x + [[0]*100] * (self.getArticleVocabSize() - len(x)))  # Padding with zeros
        self.pb.update(1)

        self.pb.set_description(f'Word2Vec - highlights')
        self.df['highlights_vectors'] = tokenized_highlights.apply(lambda x: [self.modelWord2Vec.wv[word] for word in x if word in self.modelWord2Vec.wv])
        self.df['highlights_vectors'] = self.df['highlights_vectors'].apply(lambda x: x + [[0]*100] * (self.getHighlightsVocabSize() - len(x)))  # Padding with zeros
        self.pb.update(1)
        
    def getEmbeddings(self):
        return self.modelWord2Vec.wv.vectors
    
    def getDF(self):
        return self.df

    
if __name__ == '__main__':
    df = pd.read_csv('Datasets/cnn_dailymail/test.csv')
    Process(df)
    