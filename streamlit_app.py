import streamlit as st
import pickle
import pandas as pd

st.title("KmerScan")

uploaded_file = st.file_uploader("Завантаж .pkl файл", type=["pkl"])
k = st.selectbox("Оберіть k-mer", [2, 3, 4], index=2)

if uploaded_file is not None:
    if k != 4:
        st.warning("Найкраща модель побудована для 4-mers. Оберіть 4.")
    else:
        with open("best_rf_4mers.pkl", "rb") as f:
            model = pickle.load(f)

        data = pickle.load(uploaded_file)
        df = pd.DataFrame(data)

        df["seq"] = df["seq"].apply(
            lambda x: [m.lower() for m in x if len(m) == 4 and set(m.upper()) <= set("ATCG")]
        )

        kmers = [
            a + b + c + d
            for a in "atcg"
            for b in "atcg"
            for c in "atcg"
            for d in "atcg"
        ]

        X = []
        for seq in df["seq"]:
            counts = dict.fromkeys(kmers, 0)
            for m in seq:
                counts[m] += 1
            n = len(seq) if len(seq) > 0 else 1
            counts = {key: value / n for key, value in counts.items()}
            X.append(counts)

        X_new = pd.DataFrame(X)
        X_new = X_new.reindex(columns=model.feature_names_in_, fill_value=0)

        y_pred = model.predict(X_new)
        result_percent = pd.Series(y_pred).value_counts(normalize=True) * 100
        result_percent = result_percent.reindex(model.classes_, fill_value=0)
        final_prediction = result_percent.idxmax()

        st.subheader("Фінальний прогноз для зразка")
        st.success(final_prediction)

        st.subheader("Голоси по класах (%)")
        st.bar_chart(result_percent)

        st.subheader("Прогноз для кожного зчитування")
        st.dataframe(pd.DataFrame({"Predicted": y_pred}), use_container_width=True)

        if "class" in df.columns:
            st.subheader("Мітки, наявні у завантаженому файлі")
            st.write(df["class"].astype(str).str.strip().value_counts())