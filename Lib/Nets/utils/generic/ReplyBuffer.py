import random
import torch


class ReplayBuffer:
    """
    This class implements a buffer which stores the last 50 generated images and return with a probability of 1/2
    the new one or one of the last 50
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        # it is list of batch!
        self.data = []

    def push_and_pop(self, data, prc):
        """

        :param data: batch of images
        :return: batch of images randomly drawn from last max_size generated
        """
        to_return = []
        # element is each single image in the batch
        for element in data.data:
            # insert batch dim in the first position
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                # if the buffer is not full store one or more new elemnt
                self.data.append(element)
                # return the latest stored element
                to_return.append(element)
            else:
                # when the buffer is full with 50% probability
                if random.uniform(0, 1) > prc:
                    # return a random element in the buffer
                    # the latest added element is stored in place of the returned element
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    # with 50 % probability return the latest element
                    to_return.append(element)
        # convert
        return torch.cat(to_return)
