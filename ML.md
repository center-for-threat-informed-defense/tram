# Machine Learning Model

Due to the vast amount of ATT&CK techniques (aka individual labels), we found the best approach is to construct a graph with the labels.

Each node represented a label and each edge represented an occurrence of when those two labels occurred at the same time. The weights for the edges were based on the number of examples of this. Using the node2vec algorithm, embeddings of size 32 were created based on this graph. Encoding the labels to these embeddings involved obtaining the embedding vectors of each positive label in the label array and averaging the vector. Decoding the embeddings involved using a k-nearest-neighbors classifier trained using the embeddings as input, and the original labels as output. This method can be explored more here: (https://arxiv.org/pdf/1704.03718.pdf)

This method retrieves the nearest labels to the predicted embedding, effectively finding the nearest clusters an embedding may belong to. Finally, due to the continuous nature of the embedding, a regression model was used, specifically random forest regression with TF-IDF input features.

This model performed better than all the other models in training accuracy (around 85%) and testing accuracy (around 80%). The model robustly handles techniques that don't have much data, and accurately predicts techniques in general. When analyzing the model, it was determined that most of the loss occurs from labeling a sentence as a technique when it is not a technique (false positive) and mislabeling a technique to another similar technique. 

Not only does this model perform well with the given data, but we suspect to see greater improvement as more data is aligned to ATT&CK. 