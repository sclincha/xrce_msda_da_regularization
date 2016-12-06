# xrce_msda_da_regularization
Code for ACL'16: [A Domain Adaptation Regularization for Denoising Autoencoders"](https://www.aclweb.org/anthology/P/P16/P16-2005.pdf)
Contains also implemenation of marginalized stacked denoising autoencoders [Chen et al, ICML'12](http://www.cs.cornell.edu/~kilian/papers/msdadomain.pdf).

#Abstract
Finding domain invariant features is critical for successful domain adaptation and transfer learning.
However, in the case of unsupervised adaptation, there is a significant risk of overfitting on source training data.
Recently, a regularization for domain adaptation was proposed for deep models by Ganin and Lempitsky (ICML'15).
We build on their work by suggesting a more appropriate regularization for denoising autoencoders.
Our model remains unsupervised and can be computed in a closed form. On standard text classification adaptation tasks,
our approach yields the state of the art results, with an important reduction of the learning cost.

#Dependencies
  * sklearn
  * numpy
  * scipy
