import pickle
import streamlit as st


model = pickle.load(open("milk.sav", "rb"))

st.title('Prediksi Kualitas Susu')
st.write("Lengkapi Data dibawah ini")


pH = st.number_input('Input Nilai pH Susu')
Temprature = st.number_input('Input Suhu Susu')
Taste = st.selectbox(
    "Kualitas Rasa Susu",
    [
        "Baik",
        "Buruk",
    ],
)
if Taste == "Baik":
    Taste = 1
else:
    Taste = 0
Odor = st.selectbox(
    "Kualitas Bau susu",
    [
        "Baik",
        "Buruk",
    ],
)
if Odor == "Baik":
    Odor = 1
else:
    Odor = 0
Fat = st.selectbox(
    "Tingkat Kadar Lemak Susu",
    [
        "Rendah",
        "Tinggi",
    ],
)
if Fat == "Rendah":
    Fat = 0
else:
    Fat = 1
Turbidity = st.selectbox(
    "Kekeruhan Susu",
    [
        "Rendah",
        "Tinggi",
    ],
)
if Turbidity == "Rendah":
    Turbidity = 0
else:
    Turbidity = 1
Colour = st.number_input('Input Warna Susu')

if st.button("Prediksi"):
    X = [
        [
            pH,
            Temprature,
            Taste,
            Odor,
            Fat,
            Turbidity,
            Colour,
        ]
    ]
    hasil = model.predict(X)
    if hasil[0] == 0:
        st.write("Kualitas susu buruk")
        print(hasil[0])
    elif hasil[0] == 1:
        st.write("Kualitas susu sedang")
        print(hasil[0])
    else :
        st.write("Kualitas susu baik")
        print(hasil[0])
