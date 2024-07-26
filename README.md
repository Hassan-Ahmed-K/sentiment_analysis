### Sentiment Analysis Project

#### Project Overview
The objective of this project was to build a sentiment analysis model that can classify text data, such as movie reviews or tweets, as positive or negative. This task involves natural language processing (NLP) techniques to preprocess text data and leverage deep learning models to analyze and predict sentiment. The project aimed to explore various models and ultimately achieve high accuracy in sentiment classification.

#### Dataset
The dataset used in this project consisted of text data labeled with sentiment (positive or negative). The dataset underwent preprocessing steps, including tokenization, padding, and conversion to numerical representations using word embeddings. The data was then split into training, validation, and test sets.

#### Model Architecture
The best performing model in this project had the following architecture:

1. **Embedding Layer**:
   - Converts the input text sequences into dense vectors of fixed size.
   - Input dimension: Vocabulary size
   - Output dimension: 32

2. **Bidirectional LSTM Layers**:
   - Two bidirectional LSTM (Long Short-Term Memory) layers to capture dependencies in both forward and backward directions in the text data.
   - First LSTM layer output dimension: 128
   - Second LSTM layer output dimension: 64

3. **Global Average Pooling Layer**:
   - Applied global average pooling to reduce the dimensionality of the feature maps.

4. **Dense Layers**:
   - A fully connected layer with 128 units and ReLU activation function.
   - A dropout layer to prevent overfitting (Dropout rate: 0.5).
   - Batch normalization to normalize the inputs to the next layer.
   - Another dense layer with 64 units and ReLU activation function.
   - A dropout layer to prevent overfitting (Dropout rate: 0.5).
   - Batch normalization to normalize the inputs to the next layer.
   - A dense layer with 32 units and ReLU activation function.

5. **Output Layer**:
   - A single unit with a sigmoid activation function for binary classification (positive vs. negative sentiment).

#### Model Summary
```
 Model: "sequential_6"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_6 (Embedding)     (None, 100, 32)           320000    
                                                                 
 bidirectional_3 (Bidirecti  (None, 100, 128)          49664     
 onal)                                                           
                                                                 
 bidirectional_4 (Bidirecti  (None, 100, 64)           41216     
 onal)                                                           
                                                                 
 global_average_pooling1d_1  (None, 64)                0         
  (GlobalAveragePooling1D)                                       
                                                                 
 dense_13 (Dense)            (None, 128)               8320      
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 batch_normalization_3 (Bat  (None, 128)               512       
 chNormalization)                                                
                                                                 
 dense_14 (Dense)            (None, 64)                8256      
                                                                 
 dropout_2 (Dropout)         (None, 64)                0         
                                                                 
 batch_normalization_4 (Bat  (None, 64)                256       
 chNormalization)                                                
                                                                 
 dense_15 (Dense)            (None, 32)                2080      
                                                                 
 dense_16 (Dense)            (None, 1)                 33        
                                                                 
=================================================================
Total params: 430337 (1.64 MB)
Trainable params: 429953 (1.64 MB)
Non-trainable params: 384 (1.50 KB)
_________________________________________________________________
```

#### Model Training and Performance
The model was trained using the training dataset, with a portion of the data used for validation to monitor the model's performance during training. The Adam optimizer and binary cross-entropy loss function were used for training. Dropout layers and batch normalization were included to prevent overfitting and improve generalization.

The model achieved the following results after 20 epochs:
- **Training Loss**: 0.1325
- **Training Accuracy**: 96.97%
- **Validation Loss**: 0.7069
- **Validation Accuracy**: 80.96%

The performance on the test dataset was:
- **Test Loss**: 0.6758
- **Test Accuracy**: 81.74%

#### Conclusion
This project successfully demonstrated the use of deep learning techniques for sentiment analysis. The model, incorporating embedding layers, bidirectional LSTMs, and dense layers with dropout and batch normalization, achieved good accuracy on both the training and test datasets. The achieved performance metrics highlight the model's capability to accurately classify sentiment in text data, making it a valuable tool for applications such as customer feedback analysis, social media monitoring, and movie review aggregation.
