
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="TerraMind AI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 TerraMind AI")
st.subheader("Satellite-Based Crop Stress Early Warning System")

st.info(
    "Upload satellite-derived NDVI and NDMI data "
    "to inspect vegetation changes."
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_columns = {"date", "ndvi", "ndmi"}

    if not required_columns.issubset(df.columns):
        st.error("CSV must contain date, ndvi, and ndmi columns.")

    elif df.empty:
        st.warning("The uploaded CSV is empty.")

    else:
        df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")
        df["ndmi"] = pd.to_numeric(df["ndmi"], errors="coerce")

        st.subheader("📊 Dataset Preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("Dataset Summary")
        st.dataframe(df[["ndvi", "ndmi"]].describe(),
                     use_container_width=True)

        st.subheader("📈 Vegetation Trends")
        st.caption("Demo data only — not a real satellite-based crop stress assessment.")
        st.line_chart(df.set_index("date")[["ndvi", "ndmi"]])

        valid_df = df.dropna(subset=["ndvi", "ndmi"])

        if not valid_df.empty:
            latest = valid_df.iloc[-1]

            col1, col2 = st.columns(2)
            col1.metric("Latest NDVI", f"{latest['ndvi']:.2f}")
            col2.metric("Latest NDMI", f"{latest['ndmi']:.2f}")

            if valid_df["ndvi"].iloc[-1] < valid_df["ndvi"].iloc[0]:
                st.warning(
                    "NDVI declined across the available observations. "
                    "Further investigation may be needed."
                )
            else:
                st.success(
                    "No NDVI decline detected between the first "
                    "and last valid observations."
                )
