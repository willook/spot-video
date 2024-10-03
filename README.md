# spot-video
상금 드가자

## Quick start
```
pip install -e .
python download.py {file_id}
python main.py
```

## 구조
```
spotvideo
- augment: 김현종이 만들 것
- model: 모델
- preprocess: 이미지, 신호 전처리 등
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
1. 영상 패딩(shift)이 있을 수 있는가?  
    - 현재는 뒤에만 있음, 앞에도 있다면 shift 측정 로직 필요  
2. 문제가 주어질때, 정답과 오답이 한번에 주어지는가?  
    - 한 번에 한 개만 반복적으로 주어진다면 otsu threshold 사용 불가  
3. 중간에 frame drop이 생길수 있는가
    - DTW 사용
4. 중간에 변형이 바뀔 수 있는가?
    
## 개선 사항
시간축은 동일하게 주는가?
주어진 변형 중 에코와 frame drop에 취약
속도 개선

## TODO
- drawing 부분 분리
- 로직 추가 삭제 쉽게 개선
- 변형 동영상 직접 제작 및 실험 추가

