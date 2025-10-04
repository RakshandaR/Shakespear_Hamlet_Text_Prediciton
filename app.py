import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('lstm_text_generation_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title("LSTM Text Generation")
input_text = st.text_input("Enter a seed text:")
max_sequence_len = model.input_shape[1] + 1  # +1 for the predicted word
if st.button("Predict Next Word"):
    if input_text:
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f"Predicted Next Word: {next_word}")
    else:
        st.write("Please enter some text to predict the next word.")
