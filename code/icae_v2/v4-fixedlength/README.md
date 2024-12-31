# 개선점
## 길이 고정
그럼 다음 코드를, 인코더의 경우 인풋 텍스트 토큰을 1028개로 고정하여 q mem, r mem 토큰을 각각 256개로 고정하여
stage1: 1024 + q mem 토큰 256개 = 1280
stage2: 1024 + q mem 토큰 256개 + r mem 토큰 256개 = 1536
stage3: 1024 + q mem 토큰 256개 + r mem 토큰 256개 = 1536
개로 총 토큰수를 고정하고, 디코더의 경우 아웃풋 텍스트 토큰을 512개로 고정하여
stage1: q latent 256 + 512 = 768
stage2: q latent 256 + r latent 256 + 512 = 1024
stage3: q latent 256 + r latent 256 + (prompt + output) 합 512 = 1024

## 기타
- 세그먼트 없앰
- model = torch.compile(model, mode="max-autotune")
- padding 토큰 설정
- 여기서부터 --per_device_train_batch_size 4
- evaluation 1000 step마다
- 웜업 + 코사인 어닐링
- 로라만 저장