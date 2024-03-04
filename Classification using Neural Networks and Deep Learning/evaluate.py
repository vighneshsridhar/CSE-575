### ---------- ###
"""
Please compile your own evaluation code based on the training code
to evaluate the trained network.
The function name and the inputs of the function have been predifined and please finish the remaining part.
"""

# Evaluate performs forward propogation. It iterates from 0 to batch_size and takes an input x as an image and the true
# output y from labels. Then, it goes through the layers of the network and obtains the output by executing forward on layer
# l with the input x. The loss is obtained through cross_entropy. Finally, the accuracy is determined by checking if the index
# maximizes output and the true label y match.

def evaluate(net, images, labels):
    acc = 0
    loss = 0
    batch_size = 1

    pass
    for batch_index in range(0, images.shape[0], batch_size):
        """
        Please compile your main code here.
        """
        x = images[batch_index]
        y = labels[batch_index]
        for l in range(net.lay_num):
            output = net.layers[l].forward(x)
            x = output
        loss += cross_entropy(output, y)
        if np.argmax(output) == np.argmax(y):
            acc += 1

    return acc, loss
