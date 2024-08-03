class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.node_data = node_data
        self.next = SinglyLinkedListNode


# file = open("../data/leo.txt")
# file.write('hello world')
# file.close()

# file = open("../data/rand_data.csv", 'w')
# try:
#     file.write('hello world')
# finally:
#     file.close()

#
# Python 中的 with 语句用于异常处理，封装了 try…except…finally 编码范式
with open('../data/test_run.txt') as f:
    read_data = f.read()
f.closed

import cv2

a = (1, [2, 3], 4)
a[1].append("world")
print(a)
