# Parzen Window Estimation using Gaussian Kernels for Multiclass Classification
## Procedure:

The multivariate kernel density estimator is the estimated pdf of a random sample vector. Let 𝑥 be a 𝑑-dimensional random vector with a
density function 𝑓 and let 𝑦𝑖 be a random sample drawn from 𝑓 for 𝑖 = 1, 2, … , 𝑛, where n is the number of random samples. For any real
vectors of 𝑥, the kernel density estimation is:

<p align="center">
  <img src="https://i.ibb.co/x2YqChf/1.png">
</p>

where the kernel functions is:

<p align="center">
  <img src="https://i.ibb.co/sb1CbWp/2.png">
</p>

and H is the d-by-d variance matrix. In MATLAB, I used mvksdensity function which uses a diagonal variance matrix and a product kernel. That is, H1/2 is a square diagonal matrix with the elements of vector (h1, h2, …, hd) on the main diagonal. K(x) takes the product form K(x) = k(x1)k(x2)⋯k(xd), where k(·) is a one-dimensional Gaussian kernel function. Then, the multivariate kernel density estimator becomes,

<p align="center">
  <img src="https://i.ibb.co/R7MdwbJ/3.png">
</p>

In this part, I used standard multivariate gaussian kernel where H represents the covariance matrix :

<p align="center">
  <img src="https://i.ibb.co/hccFn8h/4.png">
</p>

For each class, we can compare resulting pdfs’ multiplied by prior probabilities in natural logarithm. While 𝑔𝑖 (𝑥⃗),

                                                    𝑔𝑖 (𝑥⃗) = 𝑙𝑛 𝑃(𝑥⃗ | 𝑤𝑖 ) + 𝑙𝑛 𝑃(𝑤𝑖 )
                                                    
After iterating over all classes’ resulting if, 𝑔𝑖 (𝑥⃗) > 𝑔𝑗 (𝑥⃗), then we assign x to the class 𝑤𝑖. On a **private dataset**, recall and precision for test set:

<p align="center">
  <img src="https://i.ibb.co/tsn16VG/6.png">
</p>

## Note: 
For overall comparison of algorithms on same **private dataset** please visit [Hierarchical Clustering with SVM](https://github.com/mburakbozbey/hierarchical-clustering-with-svm) repository.
