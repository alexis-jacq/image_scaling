import numpy as np
import PIL.Image
import scipy.ndimage as nd
import random as rd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

""" input image must be 32*32*3 """


class Cnn:

    def __init__(this, structure, nb_class, image_height, image_width):

        this.layers = []

        # inner layers :
        for (nb_map_in,_),(nb_map_out,w_size) in zip(structure[:-1],structure[1:]):
            if w_size%2==0:
                w_size+=1
            w = 1-2*np.random.rand(nb_map_out, nb_map_in, w_size, w_size)
            b = 1-2*np.random.rand(nb_map_out)
            this.layers.append((w,b))

        # last layer :
        nb_map_in,_ = structure[-1]
        w = 1-2*np.random.rand(nb_class, nb_map_in, image_height, image_width)
        b = 1-2*np.random.rand(nb_class)
        this.layers.append((w,b))

    def feed(this, image):

        image = image.reshape((3,32,32)) # from cifar

        input = [image[0,:,:], image[1,:,:], image[2,:,:]] # red, blue, green

        # inner layers -> convolutions
        for w,b in this.layers[:-1]:
            output = convole(input,w,b)
            input = output

        # last layer -> complete integration
        w,b = this.layers[-1]
        output = integrate(input,w,b)

        return output

    def delta_full(this, images, labels):

        w,b = this.layers[-1]
        for

    def train(this, train_images, train_labels, nb_epoch):

        train_data_size = len(train_images)
        mini_batch_size = 5

        for epoch in range(nb_epoch):
            choice = np.random.choice(train_data_size, mini_batch_size, replace=False)
            mini_batch_image = train_images[choice]
            mini_batch_labels = train_labels[choice]

            error = this.score(mini_batch_images, mini_batch_labels)
            # compute gradient for last_layer to output (logistic)
            # compute gradient for layer to deeper_layer (tanh)
            # update w = w- alpha*delta_w

    def test(this, test_images, test_labels):
        print "ok"

    

def integrate(input,w,b):

    assert len(input)==w.shape[1]

    output = []
    for kernel, bias in zip(w,b):
        output_value = 0
        for input_map,filter in zip(input,kernel):
            output_value += np.sum(input_map*filter)
        output_value = np.tanh(output_value + bias)
        output.append(output_value)

    return output
 
def delta_full(input,output,w,b,label):

    assert len(input)==w.shape[1]

    output = []
    for kernel, bias in zip(w,b):
        output_value = 0
        for input_map,filter in zip(input,kernel):
            output_value += np.sum(input_map*filter)
        output_value = np.tanh(output_value + bias)
        output.append(output_value)

    return output
   
def convole(input,w,b):

    assert len(input)==w.shape[1]

    output = []
    for kernel, bias in zip(w,b):
        output_mat = np.zeros(input[0].shape)
        for input_map,filter in zip(input,kernel) :
            fs = filter.shape[0]
            half = (fs-1)/2
            bordered_map = mirror_border(input_map, half, half)
            for i in range(input_map.shape[0]):
                for j in range(input_map.shape[1]):
                    output_mat[i,j] += np.sum(bordered_map[i:i+fs,j:j+fs]*filter)

        output_mat = np.tanh(output_mat + bias)
        output.append(output_mat)

    return output


def mirror_border(map, lim1, lim2):
    map_dim1,map_dim2 = map.shape

    new_shape = np.array(map.shape)+2*np.array([lim1,lim2])
    result = np.ones(new_shape)

    result[lim1:-lim1, lim2:-lim2] = map 
    rev_map2 = map[:,-1::-1]
    rev_map1 = map[-1::-1,:]
    rev_map3 = map[-1::-1,-1::-1]

    # borders :
    result[lim1:-lim1, :lim2] = rev_map2[:,-lim2:]
    result[lim1:-lim1, -lim2:] = rev_map2[:,:lim2]
    result[:lim1, lim2:-lim2] = rev_map1[-lim1:,:]
    result[-lim1:, lim2:-lim2] = rev_map1[:lim1:,:]
    # corners :
    result[:lim1,:lim2] = rev_map3[:lim1,:lim2]   
    result[:lim1,-lim2:] = rev_map3[:lim1,-lim2:]
    result[-lim1:,:lim2] = rev_map3[-lim1:,:lim2]
    result[-lim1:,-lim2:] = rev_map3[-lim1:,-lim2:]

    return result


def unpickle(file): # load cifar database
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def display(image): # for cifar 
    view = image.reshape((3,32,32))
    test = np.zeros((32,32,3),dtype=np.uint8)
    for i in range(3):
        test[:,:,i]=view[i]
    img = PIL.Image.fromarray(test,'RGB')
    img.show()


if __name__ == '__main__':

    net = Cnn([(3,1),(8,5),(16,5),(32,3)], 10, 32, 32)

    dict = unpickle("cifar-10-batches-py/data_batch_1") 
    image = dict['data'][0]
    display(image)
    label = dict['labels'][0]
    print label

    test = net.feed(image)

    print test

    #plt.imshow(test, interpolation='nearest',cmap = cm.Greys_r)
    #plt.show()
