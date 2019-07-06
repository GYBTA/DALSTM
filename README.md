# DeepACLSTM
DeepACLSTM: Deep asymmetric convolutional long short-term memory neural models
 <p> <a href="https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2940-0">DeepACLSTM: deep asymmetric convolutional long short-term memory neural models for protein secondary structure prediction</a>. BMC Bioinformatics.
 <br>
Protein secondary structure (PSS) is critical to further predict the tertiary structure, understand protein function and design drugs. However, experimental techniques of PSS are time consuming and expensive, and thus it’s very urgent to develop efficient computational approaches for predicting PSS based on sequence information alone. Moreover, the feature matrix of a protein contains two dimensions: the amino-acid residue dimension and the feature vector dimension. Existing deep learning based methods have achieved remarkable performances of PSS prediction, but the methods often utilize the features from the amino-acid dimension. Thus, there is still room to improve computational methods of PSS prediction. We propose a novel deep neural network method, called DeepACLSTM, to predict 8-category PSS from protein sequence features and profile features. Our method efficiently applies asymmetric convolutional neural
networks (ACNNs) combined with bidirectional long short-term memory (BLSTM) neural networks to predict PSS, leveraging the feature vector dimension of the protein feature matrix. The evaluation metric should be weighted accuracy. You can copy the function and paste it into the keras metric.py, which can be downloaded in 'https://github.com/wentaozhu/protein-cascade-cnn-lstm'. Then compile keras, install. 

### Citation：
Please cite the following paper in your publication if it helps your research:

Guo, Y., Li, W., Wang, B., Liu, H., & Zhou, D. (2019). DeepACLSTM: deep asymmetric convolutional long short-term memory neural models for protein secondary structure prediction. BMC bioinformatics, 20(1), 341.
