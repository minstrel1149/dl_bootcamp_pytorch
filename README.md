# 딥러닝 부트캠프 with 파이토치 - 김기현 저

### 중요사항
1. Pytorch의 기초 부분을 보고 복습할 수 있도록 다시 학습 및 코드 추가
2. 책에 있는 코드를 단순히 따라하는 것이 아니라 나만의 코드로 작성
3. 변수명은 Snake Case 사용

### Chapter.2 딥러닝 소개
1. parameter Θ → y = f(x)에서 함수가 어떻게 동작할지 정해놓은 내용
2. 학습 → 함수 f가 x에서 y로 가는 관계를 배우는 것, parameter Θ를 찾아내는 과정
3. argmax → 대상 함수 f의 출력을 최대로 만드는 입력을 반환

### Chapter.3 파이토치 튜토리얼
1. Pytorch는 Autograd 제공 → 자동으로 back-propagation 계산
2. Pytorch는 broadcasting 연산 적용 → 크기가 다른 두 텐서 연산 가능
3. Pytorch 기초 함수/메서드
    * torch.Float/Long/ByteTensor(array or shape)
    * torch.from_numpy(nd.array), tensor.numpy()
    * tensor.long/float() → 타입 변환
    * tensor.size() or tensor.shape, tensor.dim()
    * tensor.sum/mean(dim)
    * tensor.view/reshape(shape)
    * tensor.squeeze(idx) → 차원의 크기가 1인 차원을 없애주는 역할
    * tensor에 대한 indexing 및 slicing 가능
    * tensor.split/chunk(size/chunks, dim)
        - chunk는 마지막까지 같은 크기로 가고, 마지막에 남는걸로 최후 처리
    * tensor.index_select(dim, index)
    * torch.concat/cat(tensor_list, dim)
    * torch.stack(tensor_list, dim)
    * tensor.expand(size) → 차원의 크기가 1인 차원을 원하는 크기로 늘려주는 역할
    * torch.randperm(n) → 딥러닝은 랜덤성에 의존하는 부분이 많기 때문에 활용
    * tensor.argmax(dim)
    * torch.topk(input, k, dim)
    * torch.sort(input, dim, descending)
    * tensor.masked_fill(mask, value) → pandas의 series.where 메서드와 유사
    * torch.ones/zeros(size), torch.ones_like/zeros_like(input)

### Chapter.4 선형 계층
1. Linear Layer는 y = W^T*x + b 형태이므로 parameter Θ는 W, b
2. Linear Layer 관련 Pytorch 기초 함수/메서드
    * torch.matmul(input, input)
    * torch.bmm(input, input) → 여러 샘플(행렬)을 동시에 병렬 계산(batch matmul)
        - 마지막 두개 차원을 제외한 다른 차원의 크기는 동일해야
    * import torch.nn as nn → nn.Module을 바탕으로 클래스 상속
        - __init__() 함수 및 forward() 함수 이용
        - nn.Parameter 클래스 활용
        - nn.Linear 클래스 활용
    * torch.cuda.FloatTensor(size), torch.device(device), tensor.to(device_obj)

### Chapter.5 손실 함수
1. loss → y_hat과 y 사이 간 차이의 크기를 더한 것
    * L1 norm, L2 norm, RMSE/MSE 등 존재
2. Loss Function 관련 Pytorch 기초 함수/메서드
    * import torch.nn.functional as F
    * F.mse_loss(y_hat, y, reduction) or nn.MSELoss()

### Chapter.6 경사하강법
1. Loss function이 최소가 되는 parameter Θ를 찾기 위한 방법의 일종 → Θ_hat = argmin L(Θ)
2. Θ ← Θ - η▽L(Θ) (단, η : Learning rate)
    * Local minimum에 빠질 우려가 있으나, 높은 차원의 공간에서는 크게 문제 되지 않음
    * η는 Hyperparameter, Adam 등의 기법 활용 가능
2. Gradient Descent 관련 Pytorch 기초 함수/메서드
    * torch.rand_like(input)
    * requires_grad 속성이 True가 되도록 설정해줘야
    * loss.backward() → Calculate gradients, 텐서의 크기는 scalar여야
3. Pytorch optim 클래스를 통해 optimizer 수행 가능
    * optimizer.zero_grad() → optimizer.step()
    * optim.SGD(model.parameters(), lr), optim.Adam(model.parameters()) 등

### Chapter.8 로지스틱 회귀
1. Activation function → Sigmoid, TanH 등
    * 로지스틱 회귀는 Linear Layer 직후 Activation Function을 넣어주어 모델 구성
    * Sigmoid의 경우 출력값의 범위는 0에서 1 사이로 고정 → 참/거짓 판단
2. Loss function은 MSE가 아니라, 주로 BCE(binary cross-entropy) 활용

### Chapter.9 심층신경망 I
1. Layer들을 깊게 쌓아올린 것 → Linear Layer를 쌓을 때 그 사이에 non-Linear function을 끼워넣는 것
    * back-propagation을 통해 효율적으로 DNN 학습, chain rule을 통해 구현
        - 미분 계산 과정이 계속해서 뒤 쪽 Layer들로 전달되는 형태
2. Regression 문제일 때는 MSE Loss function 사용
3. Gradient vanishing → DNN의 Layer가 깊어질 수록 자꾸 1보다 작은 값이 곱해져 생기는 문제
    * ReLU 혹은 Leaky ReLU를 통해 해결(ReLU는 Dead neuron 문제 발생)

### Chapter.10 확률적 경사하강법
1. SGD는 전체 데이터셋을 모델에 통과시키는 대신 랜덤 샘플링한 k개의 샘플을 모델에 통과
    * 샘플링 과정에서 비복원 추출을 수행, 랜덤 샘플링된 k개의 샘플 묶음은 mini-batch
    * 하나의 mini-batch가 통과하는 것을 iteration, 전체 샘플이 모두 통과하는 것을 epoch
    * mini-batch의 크기는 Hyperparameter
2. 이중 for-loop 진행
    * 바깥쪽 for-loop는 최대 epochs 수만큼 반복
    * 안쪽 for-loop는 mini-batch에 대하여 feed-forwarding, backpropagation, 경사하강 등 수행
    * 매 epoch 시마다 데이터 셋을 셔플링하고 mini-batch로 나눠줘야

### Chapter.11 최적화
1. Hyperparameter → 자동으로 최적화되지 않으므로 직접 실험을 통해 성능을 지켜보면서 값을 튜닝해야
    * empiricial, heuristic 방법을 통해 찾는 형태
    * 중요한 Hyperparameter와 사소한 Hyperparameter를 구분할 수 있는 능력이 필요
2. Learning rate: 대표적인 Hyperparameter, 가장 먼저 튜닝이 필요하기도
    * Θ ← Θ - η▽L(Θ) : Learning rate에 따라 학습이 진행되는 양상이 달라질 수도
    * Learning rate를 학습 내내 고정하는 것이 아니라 동적으로 가져간다면 이점 → Adaptive Learning rate로 발전
    * 요즘 가장 많이 쓰이는 알고리즘은 Adam(Adagrad + Momentum)

### Chapter.12 오버피팅을 방지하는 방법
1. 최종 목표는 generalization error를 최소화하는 것이지, training error를 최소화하는 것이 아님
    * Overfitting: training error가 generalization error에 비해 현격하게 낮아지는 현상
    * Underfitting: 모델이 충분히 데이터를 학습하지 못하여 train error가 충분히 낮지 않은 현상
    * Overfitting 여부를 확인한 이후 Overfitting을 방지하는 방향으로 모델을 개선
2. 학습 초반 Underfitting으로 시작해서 어느 순간 Overfitting으로 전환
    * Overfitting으로 전환되는 순간에 학습을 멈춘다면 가장 이상적 → Validation dataset 활용
    * 학습하는 도중에 주기적으로 모델에 넣어 loss를 구함으로써 Overfitting 여부를 확인
    * Validation dataset을 통해서는 학습을 진행하지 않으며, dataset은 랜덤하게 나눠져야
    * 학습 종료 후 가장 낮은 Validation loss를 갖는 모델을 복원
3. 결론적으로는 Train dataset, Validation dataset, Test dataset으로 나눠지는 형태

### Chapter.13 심층신경망 II
1. 이진 분류를 위한 DNN은 회귀를 위한 DNN 마지막에 Sigmoid 함수를 씌워주는 형태
2. 이진 분류의 평가 지표: Accuracy, Precision-Recall, F1 Score, ROC Curve 등