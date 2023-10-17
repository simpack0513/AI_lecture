from node import *
import matplotlib.pyplot as plt

def start():
	# x 선언
	x = [[0,0], [0,1], [1,0], [1,1], [0.5, 1], [1, 0.5], [0, 0.5], [0.5, 0], [0.5, 0.5]]
	# x = [[0,0], [0,1], [1,0], [1,1]]

	# x의 순서에 맞게 target(y)값 선언
	# y = [0,1,1,0]
	y = [0,0,0,0,0,0,0,0,1]

	# 입력값(x)의 개수 선언
	n = 2

	# 에러값을 저장할 List
	errors = []

	# 각 레이어의 노드 초기화
	D1_nodes = []
	D1_nodes.append(Node(n, "ReLU"))
	D1_nodes.append(Node(n, "ReLU"))
	D2_nodes = []
	D2_nodes.append(Node(n, "ReLU"))
	D2_nodes.append(Node(n, "ReLU"))
	F_node = Node(n, "step")

	# 반복 횟수 저장 변수
	iter = 0

	# 한 학습 사이클에서 최대 시행할 횟수
	epoch = 500

	# 처음 정확도 계산 (0~1)
	count = 0
	for i in range(len(y)):
		for j in range(len(D1_nodes)):
			D1_nodes[j].calculate_output(x[i])
		for j in range(len(D2_nodes)):
			D2_nodes[j].calculate_output(list(map(lambda x: x.output, D1_nodes)))
		F_node.calculate_output(list(map(lambda x : x.output, D2_nodes)))
		if F_node.output == y[i]:
			count += 1
	error = count / len(y)
	errors.append(1 - error)

	# 에러률이 0일 때까지 혹은 정해진 횟수만큼 반복
	print("-----LEARNING START-----")
	while iter < epoch:
		iter += 1
		# 각 x, y pair 마다 학습을 진행
		for i in range(len(y)):
			# 우선 각 퍼셉트론의 출력값을 차례로 계산
			print("y:" ,y[i] , "x: " , x[i])
			for j in range(len(D1_nodes)):
				D1_nodes[j].calculate_output(x[i])
			for j in range(len(D2_nodes)):
				D2_nodes[j].calculate_output(list(map(lambda x: x.output, D1_nodes)))
			F_node.calculate_output(list(map(lambda x : x.output, D2_nodes)))

			# Backpropagation - weight 업데이트하기
			# 마지막 퍼셉트론부터 구하고 미분값을 front_d에 담아 이전 레이어로 전달
			front_d = F_node.learning(list(map(lambda x : x.output, D2_nodes)), y[i]-F_node.output)
			# D2 레이어의 퍼셉트론의 weight를 업데이트하고 미분값을 각각 front_d1, front_d2에 담아 D1 레이어에 전달
			front_d1 = D2_nodes[0].learning(list(map(lambda x: x.output, D1_nodes)), front_d*F_node.weights[1])
			front_d2 = D2_nodes[1].learning(list(map(lambda x: x.output, D1_nodes)), front_d*F_node.weights[2])
			# 마지막 D1 레이어의 퍼셉트론 업데이트
			D1_nodes[0].learning(x[i], front_d1*D2_nodes[0].weights[1]+front_d2*D2_nodes[1].weights[1])
			D1_nodes[1].learning(x[i], front_d1*D2_nodes[0].weights[2]+front_d2*D2_nodes[1].weights[2])
			# 현재 weight 출력
			print("D1node_weights : ", end='')
			print("<P1> ", D1_nodes[0].weights, " <P2> ", D1_nodes[1].weights)
			print("<P1> ", "D2node_weights : ", end='')
			print(D2_nodes[0].weights, " <P2> ", D2_nodes[1].weights)
			print("Fnode_Weights : ", end='')
			print(F_node.weights)

		# 정확도 계산 (0~1)
		count = 0
		for i in range(len(y)):
			for j in range(len(D1_nodes)):
				D1_nodes[j].calculate_output(x[i])
			for j in range(len(D2_nodes)):
				D2_nodes[j].calculate_output(list(map(lambda x: x.output, D1_nodes)))
			F_node.calculate_output(list(map(lambda x : x.output, D2_nodes)))
			if F_node.output != y[i]:
				count += 1
		error = count / len(y)
		errors.append(error)
		print(error)
		# 에러률이 0이면 반복문 종료
		if error == 0:
			# weight 출력
			print("---Final Weight---")
			print("D1node_weights : ", end='')
			print("<P1> ", D1_nodes[0].weights, " <P2> ", D1_nodes[1].weights)
			print("D2node_weights : ", end='')
			print("<P1> ", D2_nodes[0].weights, " <P2> ", D2_nodes[1].weights)
			print("Fnode_Weights : ", end='')
			print(F_node.weights)

			# 에러률 그래프 출력
			plt.figure(1)
			plt.plot(errors, 'ro')
			plt.axis([0, iter, 0, 1])

			# D1의 퍼셉트론이 만드는 직선 출력
			plt.figure(2)
			arr = np.array(range(-3, 3))
			# x2 = (-w1x1 - w0x0) / w2
			plt.plot(arr, (-1*D1_nodes[0].weights[1]*arr - D1_nodes[0].weights[0]) / D1_nodes[0].weights[2])
			plt.plot(arr, (-1*D1_nodes[1].weights[1]*arr - D1_nodes[1].weights[0]) / D1_nodes[1].weights[2])
			plt.axis([-2, 2, -2, 2])

			# 학습 데이터 출력
			for a in range(len(y)):
				plt.plot(x[a][0], x[a][1], ('bo' if y[a] else 'ro'))

			# 0~1까지 0.05 간격으로 모든 점들의 결과가 0인지 1인지 그래프로 출력
			plt.figure(3)
			arr = []
			for a in range(0, 21):
				for b in range(0, 21):
					arr.append([a/20, b/20])
			for i in range(len(arr)):
				for j in range(len(D1_nodes)):
					D1_nodes[j].calculate_output(arr[i])
				for j in range(len(D2_nodes)):
					D2_nodes[j].calculate_output(list(map(lambda x: x.output, D1_nodes)))
				F_node.calculate_output(list(map(lambda x : x.output, D2_nodes)))
				plt.plot(arr[i][0], arr[i][1], ('bo' if F_node.output==1 else 'ro'))
			plt.axis([0, 2, 0, 2])

			plt.show()
			return 1
	return 0


# 실행구문
# 정해진 횟수까지 반복하고도 에러률이 0이 아니라면 weight를 다시 초기화하고 학습을 진행
while (start() != 1):
	pass
print("----DONE----")

