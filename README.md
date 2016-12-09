# Regularized marginalized Stacked Denoising Autoencders for Domain Adaptation
 *  Matlab Code for  Csurka, Gabriela and Chidlovskii, Boris and Clinchant, Stéphane and  Michel, Sofia, Unsupervised Domain Adaptation with Regularized Domain Instance Denoising,
 ECCV workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV), 2016 [Paper Here](http://adas.cvc.uab.es/task-cv2016/papers/0009.pdf)

 *  Python Code for ACL'16: Link to Paper: [A Domain Adaptation Regularization for Denoising Autoencoders"](https://www.aclweb.org/anthology/P/P16/P16-2005.pdf)
Contains also implemenation of marginalized stacked denoising autoencoders [Chen et al, ICML'12](http://www.cs.cornell.edu/~kilian/papers/msdadomain.pdf).

#Abstract
Finding domain invariant features is critical for successful domain adaptation and transfer learning.
However, in the case of unsupervised adaptation, there is a significant risk of overfitting on source training data.
Recently, a regularization for domain adaptation was proposed for deep models by Ganin and Lempitsky (ICML'15).
We build on their work by suggesting a more appropriate regularization for denoising autoencoders and propose to extend the marginalized denoising autoencoder (MDA)
framework with a domain regularization whose aim is to denoise both the source
and target data in such a way that the features become domain invariant and the
adaptation gets easier. The domain regularization, based either on the maximum
mean discrepancy (MMD) measure or on the domain prediction, aims to reduce
the distance between the source and the target data. We also exploit the source
class labels as another way to regularize the loss, by using a domain classifier
regularizer. Our model remains unsupervised and can be computed in a closed form. 

#Dependencies
  * sklearn
  * numpy
  * scipy
