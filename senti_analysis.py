import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Define chunk size
chunk_size = 10000

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=10000)

# Load dataset in chunks
chunk_iter = pd.read_csv("IMDB Dataset.csv", chunksize=chunk_size)

# Initialize lists to store sequences and labels
all_sequences = []
all_labels = []

# Process each chunk
for chunk in chunk_iter:
    # Prepare the data
    reviews = chunk['review'].values
    sentiments = chunk['sentiment'].values
    labels = np.where(sentiments == 'positive', 1, 0)

    # Tokenize the text data
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)

    # Pad sequences to a fixed length
    max_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Append sequences and labels to the lists
    all_sequences.append(padded_sequences)
    all_labels.append(labels)

# Concatenate sequences and labels
train_sequences = np.concatenate(all_sequences)
train_labels = np.concatenate(all_labels)

# Split the data into training and test sets
test_size = 0.2
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    train_sequences, train_labels, test_size=test_size, random_state=42)

# Define the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=max_length),
    LSTM(units=64),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_sequences, train_labels, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_sequences, test_labels)
print("Test Accuracy:", test_accuracy)

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(test_sequences, test_labels)

# Open evaluation_results.txt file in write mode
with open('evaluation_results.txt', 'w') as f:
    # Write evaluation results
    f.write('Test Loss: {}\n'.format(test_loss))
    f.write('Test Accuracy: {}\n'.format(test_accuracy))