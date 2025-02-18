import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import pandas as pd

# 안내 문구
st.write("사진을 업로드하거나 촬영하시면 유럽상, 아시아상, 아프리카상의 비율이 나옵니다!")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('keras_model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r').readlines()

# 선택 옵션: 카메라 입력 또는 파일 업로드
input_method = st.radio("이미지 입력 방식 선택", ["카메라 사용", "파일 업로드"])

if input_method == "카메라 사용":
    img_file_buffer = st.camera_input("정중앙에 사물을 위치하고 사진찍기 버튼을 누르세요")
else:
    img_file_buffer = st.file_uploader("이미지 파일 업로드", type=["png", "jpg", "jpeg"])

# 모델 입력용 배열 준비
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if img_file_buffer is not None:
    # 이미지를 RGB로 변환
    image = Image.open(img_file_buffer).convert('RGB')

    # 224 x 224 사이즈로 맞춤
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # 넘파이 배열로 변환
    image_array = np.asarray(image)

    # Normalize: [0,255] -> [-1,1]
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # 모델 입력에 맞게 data에 저장
    data[0] = normalized_image_array

    # 예측
    prediction = model.predict(data)

    # 가장 높은 확률의 클래스 인덱스
    index = np.argmax(prediction)
    # 예측 클래스 이름 (labels.txt의 i번째 줄)
    class_name = class_names[index]
    # 해당 클래스의 확률
    confidence_score = prediction[0][index]

    # 최종 예측 결과 출력
    st.write("**Class:**", class_name[2:].strip())
    st.write("**Confidence score:**", f"{confidence_score:.4f}")

    # 각 클래스별 확률(%)을 DataFrame으로 만들기
    df = pd.DataFrame({
        'Class': [label[2:].strip() for label in class_names],
        'Probability(%)': [p*100 for p in prediction[0]]
    })

    # 텍스트로 각 클래스별 확률 표시
    st.write("### 각 클래스별 예측 확률")
    for i, row in df.iterrows():
        st.write(f"{row['Class']}: {row['Probability(%)']:.2f}%")

    # 막대 그래프로 시각화
    st.bar_chart(data=df, x='Class', y='Probability(%)')
