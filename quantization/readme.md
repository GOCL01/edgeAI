

<h1> Quantization </h1>
The success of Deep Learning (DL) models has overpassed their traditional
counterpart solutions for multiple tasks (e.g. vision tasks). However, one major
challenge is the deployment of the usually overparametrized DL models into
edge devices. Local processing and edge AI have become a hot-topic in R& D
since this approach will provide solution on reducing latency, bandwidth and
communications costs and privacy concerns. Therefore a field known as model
compression has emerged, and has already provided multiple alternatives to
shrink DL models. One of the most common techniques is quantization, that
implies reducing the bit-representation of trainable parameters, which effectively
reduce memory consumption and power (accessing to memory is costly in terms
of power).
<h2> Quantization techniques </h2>

Quantization be divided into two major sub-solutions: Post-training quantiza-
tion (PTQ) and Quantization-Aware training (QAT). As their names suggest,
the first approach compresses models after training, offering the advantage of
avoiding to re-train models, however, normally accuracy is reduced. The second
approach aims to integrate quantization during training, while this approach offers the advantage 
of obtaining a higher ccuracy, it might generate longer training times.
