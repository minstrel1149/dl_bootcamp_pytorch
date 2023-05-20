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