import torch
import torch.nn as nn
import torch.optim as optim
import spacy

from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext. data import Field, BucketIterator
# from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from Utlis import calculate_bleu

# spacyGer = spacy.load("de_core_news_sm")
# spacyEng = spacy.load("en_core_web_sm")

# def tokenize_ger(text):
#     return [tok.text for tok in spacyGer.tokenizer (text)]

# def tokenize_eng (text):
#     return [tok.text for tok in spacyEng.tokenizer(text)]

# german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
# english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>",eos_token="<eos>")

from torchtext.data import Field, TabularDataset

# Define the Fields for tokenization
articles = Field(tokenize="spacy", tokenizer_language="en_core_web_sm", init_token="<sos>", eos_token="<eos>")
highlights = Field(tokenize="spacy", tokenizer_language="en_core_web_sm", init_token="<sos>", eos_token="<eos>")

# Define the fields dictionary for the TabularDataset
fields = {"article": ("src", articles), "highlights": ("trg", highlights)}

# Path to your data file
data_path = "Datasets/cnn_dailymail/train.csv"

# Create the TabularDataset
dataset = TabularDataset(
    path=data_path,
    format="csv",  
    fields=fields
)

# Split the dataset into training, validation, and test sets
train_data, valid_data, test_data = dataset.split(split_ratio=[0.7, 0.15, 0.15])

# Build the vocabulary for the Fields
articles.build_vocab(train_data, max_size=10000, min_freq=2)
highlights.build_vocab(train_data, max_size=10000, min_freq=2)

# Print some statistics
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")
print(f"Unique tokens in source vocabulary: {len(articles.vocab)}")
print(f"Unique tokens in target vocabulary: {len(highlights.vocab)}")


# trainData, validData, testData = Multi30k.splits(
#     exts=(".de", ".en"), fields=(german, english)
# )
# german.build_vocab(trainData, max_size=10000, min_freq=2) 
# english.build_vocab(trainData, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(self, 
                 embeddingSize,
                 sourceVocabSize,
                 targetVocabSize,
                 sourcePadIndex,
                 numberHeads,
                 numberEncoderLayers,
                 numberDecoderLayers,
                 forwardExpansion,
                 dropout,
                 maxLength,
                 device
                 ) -> None:
        super(Transformer, self).__init__()
        
        # Creating a map to turn words into vectors, similar to Word2Vec
        self.sourceWordEmbedding = nn.Embedding(sourceVocabSize, embeddingSize)
        
        # Creating a map to turn the position of the word into a vec
        self.sourcePositionEmbedding = nn.Embedding(maxLength, embeddingSize)
        
        # Same same, but for the target, (Need to double check to see if this needed as both the text are in english)
        self.targetWordEmbedding = nn.Embedding(targetVocabSize, embeddingSize)
        self.targetPositionEmbedding = nn.Embedding(maxLength, embeddingSize)
        
        # Set the device, GPU or CPU
        self.device = device
        
        self.transformer = nn.Transformer(
            d_model=embeddingSize,
            nhead=numberHeads,
            numberDecoderLayers=numberDecoderLayers,
            numberEncoderLayers=numberEncoderLayers,
            dim_feedforward=forwardExpansion,
            dropout=dropout
        )
        
        # Create a Linear and softmax function to turn word vectors into words
        self.fcOut = nn.Linear(embeddingSize, targetVocabSize),
        self.dropout = nn.Dropout(dropout)
        self.sourcePadIdx = sourcePadIndex
        
    def getSourceMask(self, src):
        # We are changing the shape of the mask from (srcLen, N) -> (N, srcLen)
        # it needed to be in this format for pytorch to use it :)
        sourceMask = src.transpose(0, 1) == self.sourcePadIdx
        
        return sourceMask
    
    def forward(self, source, target):
        sourceSeqLength, N = source.shape
        targetSeqLength, N = target.shape
        
        # Creating the positions used for the position embeddings
        sourcePositions = (
            torch.arrange(0, sourceSeqLength).unsqueeze(1).expand(sourceSeqLength, N)
            .to(self.device)
        )
        
        targetPositions = (
            torch.arrange(0, targetSeqLength).unsqueeze(1).expand(targetSeqLength, N)
            .to(self.device)
        )
        
        # We are combining both the word embedding with the position of the words 
        embedSource = self.dropout(
            (self.sourceWordEmbedding(source) + self.sourcePositionEmbedding(sourcePositions))
        )
        
        embedTarget = self.dropout(
            (self.targetWordEmbedding(target) + self.targetPositionEmbedding(targetPositions))
        )
        
        # Now we are creating a mask that can be used on all the text
        sourcePaddingMask = self.getSourceMask(source)
        targetMask = self.transformer.generate_square_subsequent_mask(targetSeqLength).to_dense(self.device)
        
        out = self.transformer(
            embedSource,
            embedTarget,
            src_key_padding_mask = sourcePaddingMask,
            tgt_mask = targetMask
        )
        
        out = self.fcOut(out)
        
        return out
    
def getHighlights(sentence, german, english, model, device, max_length=50):
    # Tokenize the source sentence
    tokens = german.tokenize(sentence)
    # Add <sos> and <eos> tokens
    tokens = [german.init_token] + tokens + [german.eos_token]
    # Convert tokens to indexes
    src_indexes = [german.vocab.stoi[token] for token in tokens]
    # Convert to tensor and add batch dimension
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    # Pass source sentence through the model to get the output
    with torch.no_grad():
        # Encode the source sentence
        enc_src = model.encode(src_tensor)
        # Initialize the target sentence with <sos> token
        trg_indexes = [english.vocab.stoi[english.init_token]]
        # Create a tensor for target sentence
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)
        
        # Generate target sentence one token at a time
        for _ in range(max_length):
            # Decode the target sentence
            with torch.no_grad():
                output, attention = model.decode(enc_src, trg_tensor)
            
            # Get the predicted next token
            pred_token = output.argmax(2)[-1, :].item()
            # Append predicted token to target sentence
            trg_indexes.append(pred_token)
            
            # If predicted token is <eos>, stop
            if pred_token == english.vocab.stoi[english.eos_token]:
                break
            
            # Create tensor for next input token
            trg_tensor = torch.LongTensor([pred_token]).unsqueeze(1).to(device)
        
        # Convert the indexes to tokens
        trg_tokens = [english.vocab.itos[i] for i in trg_indexes]
    
    # Remove <sos> token and return the target sentence (as a string)
    return trg_tokens[1:]

    
# Create a Training phase
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loadModel = False
saveModel = True

# Training hyperparameters
numberEpochs = 5
learningRate = 3e-4
batchSize = 32

# Model hyperparameters
sourceVocabSize, targetVocabSize = len(english.vocab), len(english.vocab)
embeddingSize = 512,
numberHeads = 8
numberEncoderLayers = 3
numberDecoderLayers = 3
dropout = 0.10
maxLength = 1500
forwardExpansion = 4,
sourcePadIndex = english.vocab.stoi['<pad>']

# Tensorboard to be fancy
writer = SummaryWriter('runs/lossPlot')
step = 0

trainIterator, validIterator, testIterator = BucketIterator.splits(
    (trainData, validData, testData),
    batch_sizes=batchSize,
    sort_within_batch = True, # Note the reason that you are sorting it is because when you are in batches it wont have to calculate extra padding size
    sort_key = lambda x: len(x.src),
    device = device
)

model = Transformer(
    embeddingSize=embeddingSize,
    sourceVocabSize=sourceVocabSize,
    targetVocabSize=targetVocabSize,
    sourcePadIndex=sourcePadIndex,
    numberHeads=numberHeads,
    numberEncoderLayers=numberEncoderLayers,
    numberDecoderLayers=numberDecoderLayers,
    forwardExpansion=forwardExpansion,
    dropout=dropout,
    maxLength=maxLength,
    device=device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learningRate)


padIndex = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = padIndex)

if loadModel:
    # Load the model
    pass
    
exampleArticle = "Ever noticed how plane seats appear to be getting smaller and smaller? With increasing numbers of people taking to the skies, some experts are questioning if having such packed out planes is putting passengers at risk. They say that the shrinking space on aeroplanes is not only uncomfortable - it's putting our health and safety in danger. More than squabbling over the arm rest, shrinking space on planes putting our health and safety in danger? This week, a U.S consumer advisory group set up by the Department of Transportation said at a public hearing that while the government is happy to set standards for animals flying on planes, it doesn't stipulate a minimum amount of space for humans. 'In a world where animals have more rights to space and food than humans,' said Charlie Leocha, consumer representative on the committee. 'It is time that the DOT and FAA take a stand for humane treatment of passengers.' But could crowding on planes lead to more serious issues than fighting for space in the overhead lockers, crashing elbows and seat back kicking? Tests conducted by the FAA use planes with a 31 inch pitch, a standard which on some airlines has decreased . Many economy seats on United Airlines have 30 inches of room, while some airlines offer as little as 28 inches . Cynthia Corbertt, a human factors researcher with the Federal Aviation Administration, that it conducts tests on how quickly passengers can leave a plane. But these tests are conducted using planes with 31 inches between each row of seats, a standard which on some airlines has decreased, reported the Detroit News. The distance between two seats from one point on a seat to the same point on the seat behind it is known as the pitch. While most airlines stick to a pitch of 31 inches or above, some fall below this. While United Airlines has 30 inches of space, Gulf Air economy seats have between 29 and 32 inches, Air Asia offers 29 inches and Spirit Airlines offers just 28 inches. British Airways has a seat pitch of 31 inches, while easyJet has 29 inches, Thomson's short haul seat pitch is 28 inches, and Virgin Atlantic's is 30-31."

for epoch in range(numberEpochs):
    print(f'[Epoch {epoch} / {numberEpochs}]')
    
    if saveModel:
        checkpoint = {
            "stateDict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        
        # save the checkpoint
        
        
        
    model.eval()
    testHighlights = getHighlights(exampleArticle, english, english, model, device)
    
    print("Key Dot Points: ")
    print(testHighlights)
    model.train()
    
    for batchIndex, batch in enumerate(trainIterator):
        inputData = batch.src.to(device)
        target = batch.trg.to(device)
        
        # forward prop
        # Note you are removing the last one because you want a shift where the output is one time step ahead of the input
        output = model(inputData, target[:-1])
        
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        
        writer.add_scalar("Training Loss", loss, global_step=step)
        step += 1
        
        
score = calculate_bleu(testData, model, german, english, device)
print(f"The BLEU score is: {score}")