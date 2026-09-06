
import streamlit as st
import pandas as pd
import joblib

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "earthquake_damage_rf_top10.pkl"
model = joblib.load(MODEL_PATH)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Earthquake Damage Prediction",
    page_icon="🏠",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏠 Earthquake Building Damage Prediction")

st.write(
    "Aplikasi untuk memprediksi tingkat kerusakan bangunan "
    "berdasarkan karakteristik struktur bangunan."
)

st.info(
    "Model menggunakan Tuned Random Forest dengan 10 fitur terpilih."
)


# ============================================================
# INPUT
# ============================================================

st.subheader("Building Characteristics")


count_floors_pre_eq = st.number_input(
    "Number of Floors",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)


height_percentage = st.number_input(
    "Height Percentage",
    min_value=1,
    max_value=100,
    value=5,
    step=1
)


age = st.number_input(
    "Building Age",
    min_value=0,
    max_value=200,
    value=20,
    step=1
)


area_percentage = st.number_input(
    "Area Percentage",
    min_value=1,
    max_value=100,
    value=8,
    step=1
)


total_superstructure = st.number_input(
    "Total Superstructure",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)


has_superstructure_mud_mortar_brick = st.selectbox(
    "Has Mud Mortar Brick Superstructure?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


has_superstructure_timber = st.selectbox(
    "Has Timber Superstructure?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


other_floor_type = st.selectbox(
    "Other Floor Type",
    options=['j', 'q', 's', 'x']
)


position = st.selectbox(
    "Building Position",
    options=['j', 'o', 's', 't']
)


superstructure_combination = st.selectbox(
    "Superstructure Combination",
    options=['adobe_mud', 'adobe_mud + bamboo', 'adobe_mud + bamboo + cement_mortar_brick', 'adobe_mud + bamboo + cement_mortar_brick + cement_mortar_stone', 'adobe_mud + bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + timber', 'adobe_mud + bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + other + rc_non_engineered + timber', 'adobe_mud + bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + other + timber', 'adobe_mud + bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + timber', 'adobe_mud + bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + timber', 'adobe_mud + bamboo + cement_mortar_brick + cement_mortar_stone + rc_engineered + rc_non_engineered + timber', 'adobe_mud + bamboo + cement_mortar_brick + cement_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + bamboo + cement_mortar_brick + cement_mortar_stone + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_brick', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + rc_engineered + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_brick + other + stone_flag + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_brick + rc_non_engineered + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_brick + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_stone + other', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_stone + other + stone_flag + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_stone + other + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_stone + rc_non_engineered + timber', 'adobe_mud + bamboo + cement_mortar_brick + mud_mortar_stone + timber', 'adobe_mud + bamboo + cement_mortar_brick + other + timber', 'adobe_mud + bamboo + cement_mortar_brick + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + other + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_engineered + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_brick + other + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_brick + rc_engineered + rc_non_engineered + stone_flag + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_brick + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_stone', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_stone + other + stone_flag + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_stone + other + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_stone + stone_flag + timber', 'adobe_mud + bamboo + cement_mortar_stone + mud_mortar_stone + timber', 'adobe_mud + bamboo + cement_mortar_stone + other + timber', 'adobe_mud + bamboo + cement_mortar_stone + timber', 'adobe_mud + bamboo + mud_mortar_brick', 'adobe_mud + bamboo + mud_mortar_brick + mud_mortar_stone', 'adobe_mud + bamboo + mud_mortar_brick + mud_mortar_stone + other', 'adobe_mud + bamboo + mud_mortar_brick + mud_mortar_stone + other + stone_flag', 'adobe_mud + bamboo + mud_mortar_brick + mud_mortar_stone + other + timber', 'adobe_mud + bamboo + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + bamboo + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + timber', 'adobe_mud + bamboo + mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'adobe_mud + bamboo + mud_mortar_brick + mud_mortar_stone + timber', 'adobe_mud + bamboo + mud_mortar_brick + other', 'adobe_mud + bamboo + mud_mortar_brick + other + stone_flag + timber', 'adobe_mud + bamboo + mud_mortar_brick + other + timber', 'adobe_mud + bamboo + mud_mortar_brick + rc_non_engineered', 'adobe_mud + bamboo + mud_mortar_brick + rc_non_engineered + timber', 'adobe_mud + bamboo + mud_mortar_brick + stone_flag + timber', 'adobe_mud + bamboo + mud_mortar_brick + timber', 'adobe_mud + bamboo + mud_mortar_stone', 'adobe_mud + bamboo + mud_mortar_stone + other', 'adobe_mud + bamboo + mud_mortar_stone + other + stone_flag + timber', 'adobe_mud + bamboo + mud_mortar_stone + other + timber', 'adobe_mud + bamboo + mud_mortar_stone + rc_non_engineered', 'adobe_mud + bamboo + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + bamboo + mud_mortar_stone + rc_non_engineered + timber', 'adobe_mud + bamboo + mud_mortar_stone + stone_flag', 'adobe_mud + bamboo + mud_mortar_stone + stone_flag + timber', 'adobe_mud + bamboo + mud_mortar_stone + timber', 'adobe_mud + bamboo + other', 'adobe_mud + bamboo + other + stone_flag + timber', 'adobe_mud + bamboo + other + timber', 'adobe_mud + bamboo + rc_non_engineered', 'adobe_mud + bamboo + rc_non_engineered + stone_flag', 'adobe_mud + bamboo + rc_non_engineered + timber', 'adobe_mud + bamboo + stone_flag + timber', 'adobe_mud + bamboo + timber', 'adobe_mud + cement_mortar_brick', 'adobe_mud + cement_mortar_brick + cement_mortar_stone', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_engineered + rc_non_engineered + stone_flag + timber', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + stone_flag', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + timber', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + other', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + rc_engineered + rc_non_engineered + timber', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + rc_non_engineered', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + timber', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + other + timber', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + stone_flag', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + timber', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + other + rc_non_engineered + stone_flag + timber', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + rc_engineered', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + rc_engineered + rc_non_engineered', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + rc_non_engineered', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + rc_non_engineered + stone_flag', 'adobe_mud + cement_mortar_brick + cement_mortar_stone + timber', 'adobe_mud + cement_mortar_brick + mud_mortar_brick', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + other', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + stone_flag', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + timber', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + other', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + other + timber', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + rc_engineered + timber', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + rc_non_engineered', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + rc_non_engineered + timber', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + stone_flag', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + stone_flag + timber', 'adobe_mud + cement_mortar_brick + mud_mortar_brick + timber', 'adobe_mud + cement_mortar_brick + mud_mortar_stone', 'adobe_mud + cement_mortar_brick + mud_mortar_stone + other', 'adobe_mud + cement_mortar_brick + mud_mortar_stone + rc_non_engineered', 'adobe_mud + cement_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag', 'adobe_mud + cement_mortar_brick + mud_mortar_stone + stone_flag', 'adobe_mud + cement_mortar_brick + mud_mortar_stone + stone_flag + timber', 'adobe_mud + cement_mortar_brick + mud_mortar_stone + timber', 'adobe_mud + cement_mortar_brick + other', 'adobe_mud + cement_mortar_brick + rc_engineered', 'adobe_mud + cement_mortar_brick + rc_non_engineered', 'adobe_mud + cement_mortar_brick + rc_non_engineered + timber', 'adobe_mud + cement_mortar_brick + timber', 'adobe_mud + cement_mortar_stone', 'adobe_mud + cement_mortar_stone + mud_mortar_brick', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + other + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + stone_flag', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + rc_engineered + rc_non_engineered', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + rc_engineered + rc_non_engineered + stone_flag + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + rc_engineered + rc_non_engineered + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + rc_non_engineered', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + rc_non_engineered + stone_flag + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_brick + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_stone', 'adobe_mud + cement_mortar_stone + mud_mortar_stone + other', 'adobe_mud + cement_mortar_stone + mud_mortar_stone + other + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_stone + rc_non_engineered', 'adobe_mud + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + stone_flag', 'adobe_mud + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_stone + stone_flag', 'adobe_mud + cement_mortar_stone + mud_mortar_stone + stone_flag + timber', 'adobe_mud + cement_mortar_stone + mud_mortar_stone + timber', 'adobe_mud + cement_mortar_stone + rc_non_engineered', 'adobe_mud + cement_mortar_stone + rc_non_engineered + stone_flag', 'adobe_mud + cement_mortar_stone + rc_non_engineered + timber', 'adobe_mud + cement_mortar_stone + stone_flag', 'adobe_mud + cement_mortar_stone + timber', 'adobe_mud + mud_mortar_brick', 'adobe_mud + mud_mortar_brick + mud_mortar_stone', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + other', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + other + stone_flag', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + other + timber', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + rc_engineered', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + rc_non_engineered', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + timber', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + stone_flag', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'adobe_mud + mud_mortar_brick + mud_mortar_stone + timber', 'adobe_mud + mud_mortar_brick + other', 'adobe_mud + mud_mortar_brick + other + rc_non_engineered', 'adobe_mud + mud_mortar_brick + other + stone_flag', 'adobe_mud + mud_mortar_brick + other + stone_flag + timber', 'adobe_mud + mud_mortar_brick + other + timber', 'adobe_mud + mud_mortar_brick + rc_engineered', 'adobe_mud + mud_mortar_brick + rc_engineered + timber', 'adobe_mud + mud_mortar_brick + rc_non_engineered', 'adobe_mud + mud_mortar_brick + rc_non_engineered + timber', 'adobe_mud + mud_mortar_brick + stone_flag', 'adobe_mud + mud_mortar_brick + stone_flag + timber', 'adobe_mud + mud_mortar_brick + timber', 'adobe_mud + mud_mortar_stone', 'adobe_mud + mud_mortar_stone + other', 'adobe_mud + mud_mortar_stone + other + stone_flag', 'adobe_mud + mud_mortar_stone + other + stone_flag + timber', 'adobe_mud + mud_mortar_stone + other + timber', 'adobe_mud + mud_mortar_stone + rc_engineered + rc_non_engineered', 'adobe_mud + mud_mortar_stone + rc_non_engineered', 'adobe_mud + mud_mortar_stone + rc_non_engineered + stone_flag', 'adobe_mud + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'adobe_mud + mud_mortar_stone + rc_non_engineered + timber', 'adobe_mud + mud_mortar_stone + stone_flag', 'adobe_mud + mud_mortar_stone + stone_flag + timber', 'adobe_mud + mud_mortar_stone + timber', 'adobe_mud + other', 'adobe_mud + other + rc_engineered + timber', 'adobe_mud + other + rc_non_engineered', 'adobe_mud + other + stone_flag', 'adobe_mud + other + stone_flag + timber', 'adobe_mud + other + timber', 'adobe_mud + rc_engineered', 'adobe_mud + rc_non_engineered', 'adobe_mud + rc_non_engineered + timber', 'adobe_mud + stone_flag', 'adobe_mud + stone_flag + timber', 'adobe_mud + timber', 'bamboo', 'bamboo + cement_mortar_brick', 'bamboo + cement_mortar_brick + cement_mortar_stone', 'bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + other + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + stone_flag + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone', 'bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + stone_flag + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + other + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + rc_engineered + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + rc_non_engineered', 'bamboo + cement_mortar_brick + cement_mortar_stone + rc_non_engineered + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + stone_flag + timber', 'bamboo + cement_mortar_brick + cement_mortar_stone + timber', 'bamboo + cement_mortar_brick + mud_mortar_brick', 'bamboo + cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + timber', 'bamboo + cement_mortar_brick + mud_mortar_brick + other + timber', 'bamboo + cement_mortar_brick + mud_mortar_brick + stone_flag + timber', 'bamboo + cement_mortar_brick + mud_mortar_brick + timber', 'bamboo + cement_mortar_brick + mud_mortar_stone', 'bamboo + cement_mortar_brick + mud_mortar_stone + other + stone_flag', 'bamboo + cement_mortar_brick + mud_mortar_stone + other + stone_flag + timber', 'bamboo + cement_mortar_brick + mud_mortar_stone + other + timber', 'bamboo + cement_mortar_brick + mud_mortar_stone + rc_non_engineered', 'bamboo + cement_mortar_brick + mud_mortar_stone + rc_non_engineered + timber', 'bamboo + cement_mortar_brick + mud_mortar_stone + stone_flag', 'bamboo + cement_mortar_brick + mud_mortar_stone + stone_flag + timber', 'bamboo + cement_mortar_brick + mud_mortar_stone + timber', 'bamboo + cement_mortar_brick + other', 'bamboo + cement_mortar_brick + other + rc_non_engineered + timber', 'bamboo + cement_mortar_brick + other + timber', 'bamboo + cement_mortar_brick + rc_non_engineered', 'bamboo + cement_mortar_brick + rc_non_engineered + timber', 'bamboo + cement_mortar_brick + stone_flag + timber', 'bamboo + cement_mortar_brick + timber', 'bamboo + cement_mortar_stone', 'bamboo + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + other + stone_flag + timber', 'bamboo + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'bamboo + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'bamboo + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + timber', 'bamboo + cement_mortar_stone + mud_mortar_brick + rc_non_engineered + timber', 'bamboo + cement_mortar_stone + mud_mortar_brick + stone_flag + timber', 'bamboo + cement_mortar_stone + mud_mortar_brick + timber', 'bamboo + cement_mortar_stone + mud_mortar_stone', 'bamboo + cement_mortar_stone + mud_mortar_stone + other', 'bamboo + cement_mortar_stone + mud_mortar_stone + other + rc_non_engineered + stone_flag + timber', 'bamboo + cement_mortar_stone + mud_mortar_stone + other + stone_flag + timber', 'bamboo + cement_mortar_stone + mud_mortar_stone + other + timber', 'bamboo + cement_mortar_stone + mud_mortar_stone + rc_engineered + rc_non_engineered', 'bamboo + cement_mortar_stone + mud_mortar_stone + rc_engineered + stone_flag + timber', 'bamboo + cement_mortar_stone + mud_mortar_stone + rc_non_engineered', 'bamboo + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'bamboo + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + timber', 'bamboo + cement_mortar_stone + mud_mortar_stone + stone_flag + timber', 'bamboo + cement_mortar_stone + mud_mortar_stone + timber', 'bamboo + cement_mortar_stone + other', 'bamboo + cement_mortar_stone + other + rc_engineered + timber', 'bamboo + cement_mortar_stone + other + rc_non_engineered + timber', 'bamboo + cement_mortar_stone + other + timber', 'bamboo + cement_mortar_stone + rc_engineered + timber', 'bamboo + cement_mortar_stone + rc_non_engineered + timber', 'bamboo + cement_mortar_stone + stone_flag + timber', 'bamboo + cement_mortar_stone + timber', 'bamboo + mud_mortar_brick', 'bamboo + mud_mortar_brick + mud_mortar_stone', 'bamboo + mud_mortar_brick + mud_mortar_stone + other + stone_flag + timber', 'bamboo + mud_mortar_brick + mud_mortar_stone + other + timber', 'bamboo + mud_mortar_brick + mud_mortar_stone + rc_non_engineered', 'bamboo + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + timber', 'bamboo + mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'bamboo + mud_mortar_brick + mud_mortar_stone + timber', 'bamboo + mud_mortar_brick + other', 'bamboo + mud_mortar_brick + other + rc_non_engineered + timber', 'bamboo + mud_mortar_brick + other + timber', 'bamboo + mud_mortar_brick + rc_non_engineered + timber', 'bamboo + mud_mortar_brick + stone_flag + timber', 'bamboo + mud_mortar_brick + timber', 'bamboo + mud_mortar_stone', 'bamboo + mud_mortar_stone + other', 'bamboo + mud_mortar_stone + other + rc_non_engineered + stone_flag + timber', 'bamboo + mud_mortar_stone + other + rc_non_engineered + timber', 'bamboo + mud_mortar_stone + other + stone_flag + timber', 'bamboo + mud_mortar_stone + other + timber', 'bamboo + mud_mortar_stone + rc_engineered + rc_non_engineered', 'bamboo + mud_mortar_stone + rc_engineered + rc_non_engineered + stone_flag + timber', 'bamboo + mud_mortar_stone + rc_engineered + stone_flag + timber', 'bamboo + mud_mortar_stone + rc_engineered + timber', 'bamboo + mud_mortar_stone + rc_non_engineered', 'bamboo + mud_mortar_stone + rc_non_engineered + stone_flag', 'bamboo + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'bamboo + mud_mortar_stone + rc_non_engineered + timber', 'bamboo + mud_mortar_stone + stone_flag', 'bamboo + mud_mortar_stone + stone_flag + timber', 'bamboo + mud_mortar_stone + timber', 'bamboo + other', 'bamboo + other + timber', 'bamboo + rc_non_engineered', 'bamboo + rc_non_engineered + stone_flag + timber', 'bamboo + rc_non_engineered + timber', 'bamboo + stone_flag', 'bamboo + stone_flag + timber', 'bamboo + timber', 'cement_mortar_brick', 'cement_mortar_brick + cement_mortar_stone', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + other + stone_flag', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + stone_flag', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + other', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + other + rc_non_engineered', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + rc_engineered', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + rc_non_engineered', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + rc_non_engineered + stone_flag', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + rc_non_engineered + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_brick + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_stone', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + other', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + other + rc_non_engineered + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + rc_engineered + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + rc_non_engineered', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + rc_non_engineered + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + stone_flag', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + stone_flag + timber', 'cement_mortar_brick + cement_mortar_stone + mud_mortar_stone + timber', 'cement_mortar_brick + cement_mortar_stone + other', 'cement_mortar_brick + cement_mortar_stone + other + rc_engineered + timber', 'cement_mortar_brick + cement_mortar_stone + other + rc_non_engineered', 'cement_mortar_brick + cement_mortar_stone + other + rc_non_engineered + timber', 'cement_mortar_brick + cement_mortar_stone + other + stone_flag', 'cement_mortar_brick + cement_mortar_stone + rc_engineered', 'cement_mortar_brick + cement_mortar_stone + rc_engineered + rc_non_engineered', 'cement_mortar_brick + cement_mortar_stone + rc_engineered + rc_non_engineered + timber', 'cement_mortar_brick + cement_mortar_stone + rc_engineered + stone_flag', 'cement_mortar_brick + cement_mortar_stone + rc_engineered + stone_flag + timber', 'cement_mortar_brick + cement_mortar_stone + rc_engineered + timber', 'cement_mortar_brick + cement_mortar_stone + rc_non_engineered', 'cement_mortar_brick + cement_mortar_stone + rc_non_engineered + stone_flag', 'cement_mortar_brick + cement_mortar_stone + rc_non_engineered + timber', 'cement_mortar_brick + cement_mortar_stone + stone_flag', 'cement_mortar_brick + cement_mortar_stone + stone_flag + timber', 'cement_mortar_brick + cement_mortar_stone + timber', 'cement_mortar_brick + mud_mortar_brick', 'cement_mortar_brick + mud_mortar_brick + mud_mortar_stone', 'cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + other + timber', 'cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + rc_non_engineered', 'cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + stone_flag', 'cement_mortar_brick + mud_mortar_brick + mud_mortar_stone + timber', 'cement_mortar_brick + mud_mortar_brick + other', 'cement_mortar_brick + mud_mortar_brick + other + rc_engineered', 'cement_mortar_brick + mud_mortar_brick + other + timber', 'cement_mortar_brick + mud_mortar_brick + rc_engineered', 'cement_mortar_brick + mud_mortar_brick + rc_non_engineered', 'cement_mortar_brick + mud_mortar_brick + rc_non_engineered + timber', 'cement_mortar_brick + mud_mortar_brick + stone_flag', 'cement_mortar_brick + mud_mortar_brick + timber', 'cement_mortar_brick + mud_mortar_stone', 'cement_mortar_brick + mud_mortar_stone + other', 'cement_mortar_brick + mud_mortar_stone + other + rc_non_engineered', 'cement_mortar_brick + mud_mortar_stone + other + rc_non_engineered + timber', 'cement_mortar_brick + mud_mortar_stone + other + timber', 'cement_mortar_brick + mud_mortar_stone + rc_engineered', 'cement_mortar_brick + mud_mortar_stone + rc_engineered + timber', 'cement_mortar_brick + mud_mortar_stone + rc_non_engineered', 'cement_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag', 'cement_mortar_brick + mud_mortar_stone + rc_non_engineered + timber', 'cement_mortar_brick + mud_mortar_stone + stone_flag', 'cement_mortar_brick + mud_mortar_stone + stone_flag + timber', 'cement_mortar_brick + mud_mortar_stone + timber', 'cement_mortar_brick + other', 'cement_mortar_brick + other + rc_engineered', 'cement_mortar_brick + other + rc_non_engineered', 'cement_mortar_brick + other + rc_non_engineered + timber', 'cement_mortar_brick + other + timber', 'cement_mortar_brick + rc_engineered', 'cement_mortar_brick + rc_engineered + rc_non_engineered', 'cement_mortar_brick + rc_engineered + stone_flag', 'cement_mortar_brick + rc_engineered + timber', 'cement_mortar_brick + rc_non_engineered', 'cement_mortar_brick + rc_non_engineered + stone_flag', 'cement_mortar_brick + rc_non_engineered + stone_flag + timber', 'cement_mortar_brick + rc_non_engineered + timber', 'cement_mortar_brick + stone_flag', 'cement_mortar_brick + stone_flag + timber', 'cement_mortar_brick + timber', 'cement_mortar_stone', 'cement_mortar_stone + mud_mortar_brick', 'cement_mortar_stone + mud_mortar_brick + mud_mortar_stone', 'cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + other', 'cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + other + timber', 'cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + rc_non_engineered + timber', 'cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + stone_flag', 'cement_mortar_stone + mud_mortar_brick + mud_mortar_stone + timber', 'cement_mortar_stone + mud_mortar_brick + rc_engineered', 'cement_mortar_stone + mud_mortar_brick + rc_non_engineered', 'cement_mortar_stone + mud_mortar_brick + rc_non_engineered + timber', 'cement_mortar_stone + mud_mortar_brick + stone_flag', 'cement_mortar_stone + mud_mortar_brick + stone_flag + timber', 'cement_mortar_stone + mud_mortar_brick + timber', 'cement_mortar_stone + mud_mortar_stone', 'cement_mortar_stone + mud_mortar_stone + other', 'cement_mortar_stone + mud_mortar_stone + other + rc_non_engineered', 'cement_mortar_stone + mud_mortar_stone + other + stone_flag + timber', 'cement_mortar_stone + mud_mortar_stone + other + timber', 'cement_mortar_stone + mud_mortar_stone + rc_engineered', 'cement_mortar_stone + mud_mortar_stone + rc_engineered + stone_flag + timber', 'cement_mortar_stone + mud_mortar_stone + rc_non_engineered', 'cement_mortar_stone + mud_mortar_stone + rc_non_engineered + stone_flag', 'cement_mortar_stone + mud_mortar_stone + rc_non_engineered + timber', 'cement_mortar_stone + mud_mortar_stone + stone_flag', 'cement_mortar_stone + mud_mortar_stone + stone_flag + timber', 'cement_mortar_stone + mud_mortar_stone + timber', 'cement_mortar_stone + other', 'cement_mortar_stone + other + rc_engineered', 'cement_mortar_stone + other + rc_non_engineered', 'cement_mortar_stone + other + rc_non_engineered + timber', 'cement_mortar_stone + other + timber', 'cement_mortar_stone + rc_engineered', 'cement_mortar_stone + rc_engineered + rc_non_engineered', 'cement_mortar_stone + rc_engineered + rc_non_engineered + stone_flag', 'cement_mortar_stone + rc_engineered + stone_flag', 'cement_mortar_stone + rc_engineered + timber', 'cement_mortar_stone + rc_non_engineered', 'cement_mortar_stone + rc_non_engineered + stone_flag', 'cement_mortar_stone + rc_non_engineered + stone_flag + timber', 'cement_mortar_stone + rc_non_engineered + timber', 'cement_mortar_stone + stone_flag', 'cement_mortar_stone + stone_flag + timber', 'cement_mortar_stone + timber', 'mud_mortar_brick', 'mud_mortar_brick + mud_mortar_stone', 'mud_mortar_brick + mud_mortar_stone + other', 'mud_mortar_brick + mud_mortar_stone + other + stone_flag', 'mud_mortar_brick + mud_mortar_stone + other + timber', 'mud_mortar_brick + mud_mortar_stone + rc_non_engineered', 'mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag', 'mud_mortar_brick + mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'mud_mortar_brick + mud_mortar_stone + rc_non_engineered + timber', 'mud_mortar_brick + mud_mortar_stone + stone_flag', 'mud_mortar_brick + mud_mortar_stone + stone_flag + timber', 'mud_mortar_brick + mud_mortar_stone + timber', 'mud_mortar_brick + other', 'mud_mortar_brick + other + rc_engineered', 'mud_mortar_brick + other + rc_engineered + timber', 'mud_mortar_brick + other + rc_non_engineered', 'mud_mortar_brick + other + rc_non_engineered + timber', 'mud_mortar_brick + other + timber', 'mud_mortar_brick + rc_engineered', 'mud_mortar_brick + rc_engineered + timber', 'mud_mortar_brick + rc_non_engineered', 'mud_mortar_brick + rc_non_engineered + stone_flag', 'mud_mortar_brick + rc_non_engineered + timber', 'mud_mortar_brick + stone_flag', 'mud_mortar_brick + stone_flag + timber', 'mud_mortar_brick + timber', 'mud_mortar_stone', 'mud_mortar_stone + other', 'mud_mortar_stone + other + rc_non_engineered', 'mud_mortar_stone + other + rc_non_engineered + stone_flag + timber', 'mud_mortar_stone + other + stone_flag', 'mud_mortar_stone + other + stone_flag + timber', 'mud_mortar_stone + other + timber', 'mud_mortar_stone + rc_engineered', 'mud_mortar_stone + rc_engineered + timber', 'mud_mortar_stone + rc_non_engineered', 'mud_mortar_stone + rc_non_engineered + stone_flag', 'mud_mortar_stone + rc_non_engineered + stone_flag + timber', 'mud_mortar_stone + rc_non_engineered + timber', 'mud_mortar_stone + stone_flag', 'mud_mortar_stone + stone_flag + timber', 'mud_mortar_stone + timber', 'other', 'other + rc_engineered', 'other + rc_engineered + rc_non_engineered', 'other + rc_non_engineered', 'other + stone_flag', 'other + stone_flag + timber', 'other + timber', 'rc_engineered', 'rc_engineered + rc_non_engineered', 'rc_engineered + stone_flag', 'rc_engineered + timber', 'rc_non_engineered', 'rc_non_engineered + stone_flag', 'rc_non_engineered + stone_flag + timber', 'rc_non_engineered + timber', 'stone_flag', 'stone_flag + timber', 'timber']
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Predict Damage Level",
    use_container_width=True
):

    input_data = pd.DataFrame([{

        "count_floors_pre_eq": count_floors_pre_eq,

        "other_floor_type": other_floor_type,

        "height_percentage": height_percentage,

        "age": age,

        "area_percentage": area_percentage,

        "superstructure_combination":
            superstructure_combination,

        "total_superstructure":
            total_superstructure,

        "has_superstructure_mud_mortar_brick":
            has_superstructure_mud_mortar_brick,

        "position":
            position,

        "has_superstructure_timber":
            has_superstructure_timber

    }])


    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]


    # Mapping label
    labels = {

        0: "Low Damage",

        1: "Medium Damage",

        2: "High Damage"

    }


    predicted_label = labels[prediction]


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Prediction Result")


    if prediction == 0:

        st.success(
            f"🟢 {predicted_label}"
        )

    elif prediction == 1:

        st.warning(
            f"🟡 {predicted_label}"
        )

    else:

        st.error(
            f"🔴 {predicted_label}"
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.subheader("Prediction Probability")


    probability_df = pd.DataFrame({

        "Damage Level": [

            "Low Damage",

            "Medium Damage",

            "High Damage"

        ],

        "Probability": probability

    })


    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    )


    st.dataframe(

        probability_df.style.format(
            {"Probability": "{:.2f}%"}
        ),

        use_container_width=True

    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Disclaimer: Prediksi ini menggunakan synthetic risk-based label "
    "dan tidak menggantikan penilaian teknis kondisi bangunan."
)
