from node import *
import matplotlib.pyplot as plt

def and_gate():
	# x1, x2 차례로 2차원배열 선언
	x = [[0,0,1,1], [0,1,0,1]]
	# x의 순서에 맞게 target(y)값 선언
	y = [0,0,0,1]
	# 입력값의 개수 선언
	n = 2
	# 노드 초기화
	one_node = Node(n)
	iter = 0

	# 에러률이 0일 때까지 반복
	print("-----AND GATE-----")
	while True:
		iter += 1
		one_node.learning(x, y)
		if one_node.errors[-1] == 0:
			break
	# 출력
	print("Weights : ", end='')
	print(one_node.weights)

	# 에러률 그래프 출력
	plt.figure(1)
	plt.plot(one_node.errors, 'ro')
	plt.axis([0, iter, 0, 1])

	# weight로 일차함수 출력
	plt.figure(2)
	x = np.array(range(0, 3))
	# x2 = (-w1x1 - w0x0) / w2
	plt.plot(x, (-1*one_node.weights[1]*x - one_node.weights[0]) / one_node.weights[2])
	plt.plot(0, 0, 'ro')
	plt.plot(0, 1, 'ro')
	plt.plot(1, 0, 'ro')
	plt.plot(1, 1, 'bo')
	plt.axis([0, 2, 0, 2])
	plt.show()


def or_gate():
	x = [[0,0,1,1], [0,1,0,1]]
	y = [0,1,1,1]
	n = 2
	iter = 0

	print("-----OR GATE-----")
	one_node = Node(n)
	while True:
		iter += 1
		one_node.learning(x, y)
		if one_node.errors[-1] == 0:
			break
	print("Weights : ", end='')
	print(one_node.weights)

	plt.figure(1)
	plt.plot(one_node.errors, 'ro')
	plt.axis([0, iter, 0, 1])

	plt.figure(2)
	x = np.array(range(0, 3))
	plt.plot(x, (-1*one_node.weights[1]*x - one_node.weights[0]) / one_node.weights[2])
	plt.plot(0, 0, 'ro')
	plt.plot(0, 1, 'bo')
	plt.plot(1, 0, 'bo')
	plt.plot(1, 1, 'bo')
	plt.axis([0, 2, 0, 2])
	plt.show()

def xor_gate():
	x = [[0,0,1,1], [0,1,0,1]]
	y = [0,1,1,0]
	n = 2
	one_node = Node(n)

	print("-----XOR GATE-----")
	# 10000번만 반복 (끝이 없기 때문에)
	for i in range(10000):
		one_node.learning(x, y)
	print("Weights : ", end='')
	print(one_node.weights)

	plt.plot(one_node.errors, 'ro')
	plt.axis([0, 10000, 0, 1])

	plt.figure(2)
	x = np.array(range(0, 3))
	plt.plot(x, (-1*one_node.weights[1]*x - one_node.weights[0]) / one_node.weights[2])
	plt.plot(0, 0, 'ro')
	plt.plot(0, 1, 'bo')
	plt.plot(1, 0, 'bo')
	plt.plot(1, 1, 'ro')
	plt.axis([0, 2, 0, 2])
	plt.show()


# 실행구문
# and_gate()
# or_gate()
xor_gate()
