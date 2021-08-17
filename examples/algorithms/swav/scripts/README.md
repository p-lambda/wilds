### Notes
1. Optimizer: I think the SwAV optimizer (SGD) is best to keep as-is. In the next line, it is used in the constructor of the LARC adaptive optimizer (which is imported from apex).
2. Learning rate follows a "linear scaling" rule according to this link: https://github.com/facebookresearch/swav/issues/37#issuecomment-720522339. I.e. for total batch size (summed over GPUs) >= 512, the LR is scaled linearly. However, for batch sizes < 512, we maintain the base LR at 0.6. We use a final learning rate 1000x smaller than the base; this is because each of the base/final pairs that the paper presents are a factor of 1000x different, but they don't explicit discuss it.
3. The weight decay is always set to 1e-6 in the paper, so we kept it constant without tinkering with it.
4. If the resolution for all images is 224x224, the multi-cropping strategy doesn't need to be edited.

### Hyperparameters that definitely need to be tuned
1. Number of prototypes. The author says that 10x the number of classes, or approx. the number of subpopulations, should be good. For ImageNet, they use 3000, but at the bottom of pg. 19 of the paper they hypothesize that the number of prototypes has little influence on performance as long as there are "enough" prototypes.
2. Queue start. We tried introducing either after 60 epochs or not at all, and for DomainNet removing the queue entirely helped the most. Generally, you want the queue to have "good" features, so if you think that after 60 epochs the network has learned "good" features, then maybe use the queue. Our method for tuning this was running SwAV twice, once with and once without, and checking the loss curves / performance. (if the loss dips and then rises again when queue is introduced, it's probably not working.)

### Hyperparameters that might need to be tuned
1. Epsilon. We used 0.03 throughout, as suggested in the "common issues" section of SwAV's README. We didn't tweak it afterwards, so it's unclear whether this helped.
2. Number of epochs. We kept it at 400 epochs, since there are diminished benefits of running for 800 epochs.

