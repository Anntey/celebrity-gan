# Celebrity GAN

__Data__: CelebA dataset[<sup> [1]</sup>](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (202 599 images of celebrities)

__Task__: generate fake images (64x64) of people

__Evaluation__: subjective

__Solution__: RaLSGAN[<sup> [2]</sup>](https://arxiv.org/abs/1807.00734) (Relativistic Average Least Squares GAN)

__Success__: Decent. The network had obvious problems with smiling (teeth visible vs. not visible), clothing (especially women's clothing, covered chest/shoulders vs not covered). The images were quite heterogenous which a simple center crop can't resolve. Results could be improved by (1) cropping just the faces with a detector or available metadata. This would solve many of the aforementioned problems.

![](example_output.png)
