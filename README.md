![alt text](https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/depthwise-separable-convolution.png "Normal Convolution vs Depthwise seperable")
Comparison of a normal convolution and a depthwise separable convolution. **a)** Standard convolution with a 3x3 kernel and 3 input channels. The projection of one value is shown from the 3x3x3 (dark blue) input values to 6 colorful outputs which would be 6 output channels. **b)** Depthwise separable convolution with a 3x3 kernel and 3 input channels. First a depthwise convolution projects 3x3 pixels of each input channel to one corresponding output pixel (matching colors). Then a pointwise convolution uses these 3 output pixels to determine the 6 final output pixels.

# Depthwise seperable convolution using PyTorch
The depthwise convolution unlike the standard convolution acts only on a single channel of the input map at a time. So for each channel, we need to compute ` W * H * 1 * K * K ` FLOPs. As we have `C channels`, this sums to `W * H * C * K * K FLOPs` . The result is a map of `C * W * H` just like our input map.`

## Total operations
Now let’s add the depthwise and the pointwise convolution together: `W * H * C * K * K + W * H * C * O = W * H * C * (O + K * K)`. Note the difference to the standard convolution: here we have `(O + K * K)` as a factor instead of `O * K * K` in the standard convolution. Let’s say `K = 3` and `O = 64`, then `(O + K * K) = 73`, but `O * K * K = 576`. So for this example, the standard convolution has about 8 times as many multiplications to be calculated!

## Total parameters
It’s not only less operations, but also less parameters. <br>

The standard convolution has `C * K * K * O` learnable parameters as the kernel `K * K` needs to be learned for all the input channels `C` and the output channels `O`.<br>

In contrast, depthwise separable convolution has `C * K * K` learnable parameters for the depthwise convolution and `C * O` parameters for the pointwise convolution. <br>

Together, that’s `C * (K * K + O)` parameters which is again about 8 times less than the standard convolution (for K = 3 and O = 64).

# Implementation in PyTorch
checkout the code in `depthwise-seperable-convoulution.py`

As you can see it’s super easy to implement and can save you a lot of parameters. You simply change the standard convolution to have the same number of out_channels as in_channels (here: 10) and also add the groups parameter which you set to the same value as well. This takes care of our spatial interactions and the groups separates the channels from each other. Then the pointwise convolution is just a convolution mapping the in_channels to the out_channels we had before (here: 32) using a kernel of size 1. To bind them together, you can use the torch.nn.Sequential, so they are executed one after the other as a bundled module.

Note that for simplicity, I set bias=False for all convolutions here as I didn’t mention the bias in any of the formulas in this post.

# References
A huge thanks to this [blogpost](https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/) for providing all the resources. 
