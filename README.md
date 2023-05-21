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

### Chapter.7 선형 회귀
1. 