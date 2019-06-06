import numpy as np
import pickle as pkl


def function_generator(init_num):
    seq = np.array([], dtype='int')
    n = init_num
    seq = np.append(seq, n)
    while True:
        if ((n%2)==0):
            next_number = n/2
            next_number = np.asarray(next_number, dtype='int')
            seq = np.append(seq, next_number)
            if next_number==1:
                break
        else:
            next_number = (3*n)+1
            next_number = np.asarray(next_number, dtype='int')
            seq = np.append(seq, next_number)
        n = next_number
    return seq


output_seq_data = []
output_seq_length = []
x_train = []
y_train = []
num = 0
for n in range(0,10000):
    sequence = function_generator(n+1)
    seq_len = len(sequence)
    x_training = sequence[:(seq_len-1)]
    x_training = np.array(x_training, dtype='int')
    y_training = sequence[1:seq_len]
    y_training = np.array(y_training, dtype='int')
    output_seq_data.append(sequence)
    output_seq_length.append(seq_len)
    x_train.append(x_training)
    y_train.append(y_training)


output_seq_data = np.asarray(output_seq_data)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print(y_train[26])

output_seq_length = np.asarray(output_seq_length)
max_length = output_seq_length.max()
# print(max_length)

# print(x_train[26])

# np.save('generated_data.npy', gen_data)
# np.save('x_train.npy', x_train)
# np.save('y_train.npy', y_train)
