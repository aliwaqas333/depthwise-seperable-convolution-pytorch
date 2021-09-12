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
