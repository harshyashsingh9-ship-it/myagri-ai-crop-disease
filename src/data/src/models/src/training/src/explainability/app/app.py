import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="MyAgri AI",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 MyAgri AI")
st.subheader("AI-Based Crop Disease Prediction")

st.write(
    "Upload a crop leaf image for AI-assisted "
    "disease screening."
)

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    if st.button("Analyze Crop"):

        st.info(
            "Model inference will be connected "
            "after model training."
        )

st.warning(
    "This system provides AI-assisted preliminary "
    "screening and should not replace qualified "
    "agricultural advice."
)
