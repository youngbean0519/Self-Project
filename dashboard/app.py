import altair as alt
import streamlit as st

MODELS = {
    "ensemble": "Ensemble (KoBERT 25% + KoELECTRA 75%)",
    "lr": "Logistic Regression",
    "svm": "Linear SVM",
    "textcnn": "TextCNN",
    "bilstm": "BiLSTM",
    "kobert": "KoBERT",
    "koelectra": "KoELECTRA",
}


def show_results(results, name):
    is_svm = name == "svm"
    score_column = "결정 점수" if is_svm else "모델 점수"
    score_format = ".3f" if is_svm else ".1f"
    score_suffix = "" if is_svm else "점"

    display = results.rename(columns={"선택": "예측 여부"}).copy()
    if not is_svm:
        display[score_column] = display[score_column] * 100

    top = display.head(5).copy()
    top["점수 표시"] = top[score_column].map(
        lambda score: f"{score:{score_format}}{score_suffix}"
    )
    top["막대 라벨"] = top["점수 표시"].where(
        ~top["예측 여부"], top["점수 표시"] + " ✓"
    )

    chart = alt.Chart(top).encode(
        x=alt.X(
            f"{score_column}:Q",
            title=score_column if is_svm else "모델 점수 (100점 만점)",
            scale=alt.Scale(zero=True, padding=68) if is_svm else alt.Scale(domain=[0, 100]),
        ),
        y=alt.Y("감정:N", sort=top["감정"].tolist(), title=None),
    )
    bars = chart.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.condition(
            alt.datum["예측 여부"], alt.value("#2563EB"), alt.value("#94A3B8")
        ),
        tooltip=[
            alt.Tooltip("감정:N"),
            alt.Tooltip("점수 표시:N", title=score_column),
        ],
    )
    labels = chart.encode(text="막대 라벨:N")
    positive = labels.transform_filter(alt.datum[score_column] >= 0)
    negative = labels.transform_filter(alt.datum[score_column] < 0)

    st.subheader("상위 감정 점수")
    st.altair_chart(
        (
            bars
            + positive.mark_text(align="left", dx=6)
            + negative.mark_text(align="right", dx=-6)
        ).properties(height=200, padding={"left": 5, "right": 68}),
    )

    selected = display.loc[display["예측 여부"], "감정"].tolist()
    summary = ", ".join(selected) if selected else "없음"
    st.caption(f"예측 감정 · {summary}")

    with st.expander(f"전체 {len(display)}개 감정 점수"):
        st.dataframe(
            display,
            hide_index=True,
            column_config={
                score_column: st.column_config.NumberColumn(
                    format="%.3f" if is_svm else "%.1f점",
                ),
                "예측 여부": st.column_config.CheckboxColumn(),
            },
        )


def main():
    st.set_page_config(page_title="한국어 문장 감정 분석")
    st.title("한국어 문장 감정 분석")

    with st.form("emotion_analysis"):
        text = st.text_area(
            "분석할 문장",
            placeholder="한국어 문장을 입력해주세요.",
        )
        name = st.selectbox("분석 모델", list(MODELS), format_func=MODELS.get)
        submitted = st.form_submit_button("분석", type="primary")

    if not submitted:
        return
    if not text.strip():
        st.warning("분석할 문장을 입력해 주세요.")
        return

    try:
        with st.spinner(f"{MODELS[name]} 분석 중..."):
            from inference import predict
            results = predict(text, name)
    except Exception as exc:
        st.error(f"{MODELS[name]} 분석에 실패했습니다.")
        with st.expander("오류 상세"):
            st.exception(exc)
        return

    show_results(results, name)


if __name__ == "__main__":
    main()
