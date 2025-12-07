"""
프로젝트: 시군구별 위험 점수 지도 시각화 (2019~2023)

필요한 원본 데이터 파일 (같은 폴더 or 하위 폴더에 존재해야 함)
- tidy_with_derived_v3.xlsx   : 시도/시군구별 위험 점수 (2019~2023)
- TL_SCCO_SIG.json            : 시군구 경계 (행정구역 폴리곤)
- TL_SCCO_CTPRVN.json         : 시도 경계 (시도 코드 → 시도명 매핑용)

이 스크립트가 하는 일 (요약)
1) 위 3개 파일을 읽어온다.
2) 엑셀 쪽 시군구 이름을 정리해서 "지역키_normal" 을 만든다.
3) 시군구 경계에도 같은 방식으로 "지역키_normal" 을 만든다.
4) 창원/전주시처럼 "구 단위는 있지만 엑셀에는 시 단위만 있는" 지역을 수동으로 매핑한다.
5) 그 외 남는 결측값은 시도 평균 → 전국 평균으로 채워 넣는다.
6) 최종 GeoDataFrame(gdf)을 이용해서 연도별/변화량 지도를 그린다.
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 분석에 사용할 연도 목록
YEARS = [2019, 2020, 2021, 2022, 2023]
RISK_COLS = [f"risk_score_{y}" for y in YEARS]


# -----------------------------------------------------------
# 1. 파일 경로 찾기
# -----------------------------------------------------------

def find_paths(base="."):
    """
    현재 디렉터리(base) 아래에서
    필요한 3개 파일의 경로를 자동으로 찾아서 반환한다.
    - return: (엑셀 경로, 시군구 경계 경로, 시도 경계 경로)
    """
    base = Path(base)
    excel = next(base.rglob("tidy_with_derived_v3.xlsx"))
    sig   = next(base.rglob("TL_SCCO_SIG.json"))
    ctp   = next(base.rglob("TL_SCCO_CTPRVN.json"))
    return excel, sig, ctp


# -----------------------------------------------------------
# 2. 원본 데이터 읽기
# -----------------------------------------------------------

def load_raw_data():
    """
    3개 원본 파일을 읽어서 (df, sig, ctp)를 반환한다.
    - df  : 엑셀 (시도/시군구별 위험 점수)
    - sig : 시군구 경계 (GeoDataFrame)
    - ctp : 시도 경계 (GeoDataFrame)
    """
    excel_path, sig_path, ctp_path = find_paths()
    print("[INFO] excel:", excel_path)
    print("[INFO] sig  :", sig_path)
    print("[INFO] ctp  :", ctp_path)

    df  = pd.read_excel(excel_path)
    sig = gpd.read_file(sig_path)
    ctp = gpd.read_file(ctp_path)
    return df, sig, ctp


# -----------------------------------------------------------
# 3. 엑셀 데이터 전처리
# -----------------------------------------------------------

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    엑셀 데이터(df)를 전처리해서 반환한다.

    주요 작업
    - '전국' 행 제거
    - '시도 == 시군구' 인 시도 합계(세종 제외) 제거
    - 시군구 이름에서 공백/‘통합’ 제거 → 시군구_normal
    - 시도 + 시군구_normal → 지역키_normal 생성
    """
    # 3-1. 전국 / 시도 합계(세종 제외) 제거
    mask_keep = (df["시도"] != "전국") & (
        ~((df["시도"] == df["시군구"]) & (df["시도"] != "세종특별자치시"))
    )
    df = df[mask_keep].copy()

    # 3-2. 양쪽 공백 제거
    df["시도"] = df["시도"].str.strip()
    df["시군구"] = df["시군구"].str.strip()

    # 3-3. 시군구 이름 정규화 (공백 제거 + '통합' 제거)
    df["시군구_normal"] = (
        df["시군구"]
        .str.replace(" ", "", regex=False)
        .str.replace("통합", "", regex=False)
    )

    # 3-4. 엑셀 쪽 지역 키 (정규화 버전)
    df["지역키_normal"] = df["시도"] + " " + df["시군구_normal"]

    return df


# -----------------------------------------------------------
# 4. 시군구/시도 경계 전처리
# -----------------------------------------------------------

def preprocess_shapes(sig: gpd.GeoDataFrame,
                      ctp: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    시군구 경계(sig)와 시도 경계(ctp)를 이용해
    시군구 GeoDataFrame 에 시도 이름과 지역키_normal 을 생성한다.

    - TL_SCCO_SIG 의 SIG_CD 앞 2자리 = 시도 코드
    - TL_SCCO_CTPRVN 의 CTPRVN_CD 를 이용해 시도 이름 매핑
    """
    sig = sig.copy()

    # 4-1. 시도 코드 → 시도명 매핑 딕셔너리 생성
    ctprvn_map = dict(zip(ctp["CTPRVN_CD"], ctp["CTP_KOR_NM"]))

    # 4-2. SIG_CD 앞 2자리로 시도 코드 추출
    sig["CTPRVN_CD"] = sig["SIG_CD"].astype(str).str[:2]
    sig["시도"] = sig["CTPRVN_CD"].map(ctprvn_map)

    # 4-3. 시군구 이름 정리 (공백 제거 전 원본/정규화 둘 다 저장)
    sig["시군구"] = sig["SIG_KOR_NM"].str.strip()
    sig["시군구_normal"] = sig["시군구"].str.replace(" ", "", regex=False)

    # 4-4. 지도 쪽 지역키 (정규화 버전)
    sig["지역키_normal"] = sig["시도"] + " " + sig["시군구_normal"]

    return sig


# -----------------------------------------------------------
# 5. 엑셀 + 시군구 경계 병합
#    (창원/전주 구단위는 시단위 값으로 수동 매핑)
# -----------------------------------------------------------

def merge_with_manual_mapping(df: pd.DataFrame,
                              sig: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    엑셀(df)과 시군구 경계(sig)를 지역키_normal 기준으로 병합한다.
    다만, 아래 7개 구는 엑셀에 '시 전체' 값만 있어서 수동 매핑을 해준다.

    - 경상남도: 창원시마산합포구, 창원시마산회원구, 창원시성산구,
                창원시의창구, 창원시진해구
    - 전라북도: 전주시덕진구, 전주시완산구
    """

    sig = sig.copy()

    # 5-1. 수동 매핑 사전 정의
    manual_map = {
        "경상남도 창원시마산합포구": "경상남도 창원시마산",
        "경상남도 창원시마산회원구": "경상남도 창원시마산",
        "경상남도 창원시성산구":   "경상남도 창원시창원",
        "경상남도 창원시의창구":   "경상남도 창원시창원",
        "경상남도 창원시진해구":   "경상남도 창원시진해",
        "전라북도 전주시덕진구":   "전라북도 전주시",
        "전라북도 전주시완산구":   "전라북도 전주시",
    }

    # 5-2. 지도 쪽 지역키_normal을 엑셀 키로 교체한 버전 생성
    #      (수동 매핑이 필요한 곳만 값이 바뀜)
    sig["지역키_normal_mapped"] = sig["지역키_normal"].replace(manual_map)

    # 5-3. 실제 병합 수행
    gdf = sig.merge(
        df[["지역키_normal"] + RISK_COLS],
        left_on="지역키_normal_mapped",
        right_on="지역키_normal",
        how="left",
    )

    return gdf


# -----------------------------------------------------------
# 6. 남은 결측 값 보정 (시도 평균 → 전국 평균)
# -----------------------------------------------------------

def fill_missing_with_means(df: pd.DataFrame,
                            gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    병합 이후에도 남아 있는 NaN 을 보정한다.

    순서
    1) 같은 시도 안에서의 평균 값으로 채움
    2) 그래도 남는 NaN 은 전국 평균 값으로 채움
    (시각화 목적이므로, 실제 통계 분석에서는 더 엄밀한 방법이 필요할 수 있음)
    """
    gdf = gdf.copy()

    # 6-1. 엑셀 기준 시도별 평균, 전국 평균 계산
    sido_means = df.groupby("시도")[RISK_COLS].mean()
    nat_means  = df[RISK_COLS].mean()

    # 6-2. 각 연도별로 시도 평균 → 전국 평균 순서로 채우기
    for col in RISK_COLS:
        # 시도 평균으로 채움
        gdf[col] = gdf[col].fillna(gdf["시도"].map(sido_means[col]))
        # 그래도 NaN 이면 전국 평균으로 채움
        gdf[col] = gdf[col].fillna(nat_means[col])

    print("[INFO] 결측 개수 (모두 0이면 정상):")
    print(gdf[RISK_COLS].isna().sum())

    return gdf


# -----------------------------------------------------------
# 7. 시각화 함수들
# -----------------------------------------------------------

def plot_single_year(gdf: gpd.GeoDataFrame, year: int):
    """
    특정 연도(year)의 시군구별 위험 점수 지도를 그리는 함수.
    보고서에 한 장짜리 지도 넣을 때 사용.
    """
    col = f"risk_score_{year}"

    fig, ax = plt.subplots(figsize=(4, 6))
    gdf.plot(
        column=col,
        cmap="OrRd",
        legend=True,
        linewidth=0.2,
        edgecolor="black",
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title(f"{year}년 시군구별 위험 점수", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_years_row(gdf: gpd.GeoDataFrame, years=YEARS):
    """
    여러 연도를 한 줄에 나란히 비교해 보는 함수.
    - 모든 연도에 대해 색 범위를 '공통'으로 맞춰서
      연도별 진하기 차이를 직관적으로 볼 수 있게 함.
    """
    all_vals = pd.concat([gdf[f"risk_score_{y}"] for y in years])
    vmin, vmax = all_vals.min(), all_vals.max()
    print("[INFO] 공통 색 범위:", vmin, "→", vmax)

    fig, axes = plt.subplots(1, len(years), figsize=(4 * len(years), 6))

    for ax, y in zip(axes, years):
        col = f"risk_score_{y}"
        gdf.plot(
            column=col,
            cmap="OrRd",
            vmin=vmin,
            vmax=vmax,
            linewidth=0.2,
            edgecolor="black",
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(f"{y}년")

    fig.suptitle("연도별 시군구 위험 점수 비교", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_change(gdf: gpd.GeoDataFrame,
                year_from: int = 2019,
                year_to: int = 2023):
    """
    두 연도 사이의 변화량(악화/개선)을 보여주는 지도.
    - 빨간색: 위험 점수가 증가(악화)
    - 파란색: 위험 점수가 감소(개선)
    """
    col_from = f"risk_score_{year_from}"
    col_to   = f"risk_score_{year_to}"
    diff_col = f"risk_diff_{year_to}_{year_from}"

    gdf[diff_col] = gdf[col_to] - gdf[col_from]

    fig, ax = plt.subplots(figsize=(4, 6))
    gdf.plot(
        column=diff_col,
        cmap="coolwarm",
        legend=True,
        linewidth=0.2,
        edgecolor="black",
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title(f"{year_from}→{year_to} 위험 점수 변화", fontsize=14)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# 8. 메인 실행부
# -----------------------------------------------------------

def main():
    """
    이 스크립트를 직접 실행했을 때 동작하는 기본 파이프라인.
    - 3개 파일 읽기 → 전처리 → 병합 → 결측 보정 → 시각화 예시 3개 출력
    """
    # 1) 원본 데이터 읽기
    df_raw, sig_raw, ctp_raw = load_raw_data()

    # 2) 전처리
    df  = preprocess_df(df_raw)
    sig = preprocess_shapes(sig_raw, ctp_raw)

    # 3) 병합 (창원/전주 수동 매핑 포함)
    gdf = merge_with_manual_mapping(df, sig)

    # 4) 남은 결측 값 보정
    gdf = fill_missing_with_means(df, gdf)

    # 5) 시각화 예시
    plot_single_year(gdf, 2023)   # 2023년 지도 1장
    plot_years_row(gdf, YEARS)    # 2019~2023 5개년 비교
    plot_change(gdf, 2019, 2023)  # 2019→2023 변화 지도


if __name__ == "__main__":
    main()
