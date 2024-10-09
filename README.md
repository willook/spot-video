# spot-video
상금 드가자

## Quick start
```
pip install -e .
python download.py
python generate_data.py
python main.py
```
## 성능
### markcloud 데이터 성능 - 10.09
```
python main.py --same_length --use_cache --threshold 0.85 --data_dir data/markcloud --log_dir log_dir/markcloud/
```
- Average Accuracy: 0.967  
- Average F1: 0.979  
### youtube 데이터 성능 - 10.09
```
python main.py --same_length --use_cache --threshold 0.85 --data_dir data/youtube --log_dir log_dir/youtube/
```
- Average Accuracy: 0.998  
- Average F1: 0.909  

## 구조
```
spotvideo
- augment: 김현종이 만들 것
- model: 모델
- preprocess: 이미지, 신호 전처리 등
- type: dataset, result 등 custom data class
- util: 잡다
```

## 전체 알고리즘
1. 이미지 전처리  
    A. gray-scale 이미지로 전환 - 색 변조 대처  
    B. 이미지 가우시안 필터 적용 - 이미지 노이즈 대처  
2. 영상 신호 추출  
    A. 소수 값 keyframe interval 설정 - 속도 향상 및 frame drop 대처  
    B. 이미지 변화량 추출 - 이미지 변형에 무관한 feature  
3. 신호 후처리  
    A. 5%의 이상치 제거  
    B. moving average  
    C. normalize  
4. original video와 유사도 측정  
    A. cosine similarity  
5. threshold 기반 분류  
    A. otsu threshold 이용  

## 문의 사항
1. 변형 영상 또는 노이즈 영상 길이가 서로 다를 수 있는가?  
    - e.g. 영상 패딩, 현재는 뒤에만 있으나 앞에도 있다면 shift 측정 로직 필요
    - e.g. frame drop이 실제 프레임 개수를 바꾸도록 할 수 있음
    - e.g. 노이즈 영상의 길이가 다를 수 있음
    - 이 경우 correlation 또는 DTW 알고리즘 필요
2. 문제가 주어질때, 정답과 오답이 한번에 주어지는가?  
    - 한 번에 한 개만 반복적으로 주어진다면 otsu threshold 사용 불가  
3. 변형은 고정적인 것인가?
    - 영상이 이동하거나 빙글빙글 돌 수 있는지
      
## 개선 사항
주어진 변형 중 에코에 취약  
속도 개선  

## TODO
- [ ] drawing 부분 분리
- [ ] 로직 추가 삭제 쉽게 개선
- [ ] 변형 동영상 직접 제작 및 실험 추가
- [ ] from scipy.spatial import procrustes
- [ ] 선형 stretching, shift 변형 찾기

