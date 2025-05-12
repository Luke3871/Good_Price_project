import pandas as pd
import geopandas as gpd
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

file_paths = {
    "상권변화지표": "C:/Users/lluke/OneDrive/바탕 화면/SDC Data/서울시 상권분석서비스(상권변화지표-행정동).csv",
    "길단위인구": "C:/Users/lluke/OneDrive/바탕 화면/SDC Data/서울시 상권분석서비스(길단위인구-행정동).csv",
    "집객시설": "C:/Users/lluke/OneDrive/바탕 화면/SDC Data/서울시 상권분석서비스(집객시설-행정동).csv",
    "직장인구": "C:/Users/lluke/OneDrive/바탕 화면/SDC Data/서울시 상권분석서비스(직장인구-행정동).csv",
    "점포": "C:/Users/lluke/OneDrive/바탕 화면/SDC Data/서울시 상권분석서비스(점포-행정동).csv",
    "아파트": "C:/Users/lluke/OneDrive/바탕 화면/SDC Data/서울시 상권분석서비스(아파트-행정동).csv",
    "소득소비": "C:/Users/lluke/OneDrive/바탕 화면/SDC Data/서울시 상권분석서비스(소득소비-행정동).csv"
}

# 분석 기간 2024
quarters = [20241, 20242, 20243, 20244]

df_서울시_행정동_통합 = {
    name: pd.read_csv(path, encoding='cp949').head()
    for name, path in file_paths.items()
}


# 데이터 필터링 함수 정의 평균/총합/존재하는 마지막 값값
def filtering (path, cols, method='mean'):
    df = pd.read_csv(path, encoding='cp949')
    df = df[df["기준_년분기_코드"].isin(quarters)]
    if method == 'mean':
        df = df.groupby("행정동_코드")[cols].mean().reset_index()
    elif method == 'sum':
        df = df.groupby("행정동_코드")[cols].sum().reset_index()
    elif method == 'last':
        df = df.sort_values("기준_년분기_코드").drop_duplicates("행정동_코드", keep='last')[["행정동_코드"] + cols]
    return df

"""-------------------------------------------------------------------------------------
1. Pre-processing & merge
csv 파일별 필요한 전처리 & 병합합
-------------------------------------------------------------------------------------"""

# 1. 상권변화지표
df_상권변화 = filtering(file_paths["상권변화지표"], ["운영_영업_개월_평균", "폐업_영업_개월_평균"])

# 2. 소득소비
df_소득소비 = filtering(file_paths["소득소비"], ["월_평균_소득_금액", "지출_총금액", "음식_지출_총금액"])

# 3. 아파트 평균 시가
df_아파트 = pd.read_csv(file_paths["아파트"], encoding='cp949')
df_아파트 = df_아파트[df_아파트["기준_년분기_코드"].isin(quarters)]
df_아파트 = df_아파트.sort_values("기준_년분기_코드").drop_duplicates("행정동_코드", keep="last")
df_아파트 = df_아파트[["행정동_코드", "아파트_평균_시가"]]

# 4. 음식점 수 (점포)
df_점포 = pd.read_csv(file_paths["점포"], encoding='cp949')
df_점포 = df_점포[df_점포["기준_년분기_코드"].isin(quarters)]
food_keywords = ["한식", "중식", "일식", "분식", "양식", "패스트푸드", "치킨", "족발", "도시락", "뷔페", "기타 외식", "외식"]
df_food = df_점포[df_점포["서비스_업종_코드_명"].str.contains('|'.join(food_keywords))]
df_음식점수 = df_food.groupby("행정동_코드")["점포_수"].sum().reset_index()
df_음식점수.rename(columns={"점포_수": "음식점_수"}, inplace=True)

# 5. 직장인구 (YB/OB, 성별 포함)
df_직장인구 = pd.read_csv(file_paths["직장인구"], encoding='cp949')
df_직장인구 = df_직장인구[df_직장인구["기준_년분기_코드"].isin(quarters)]

YB_cols_male = [col for col in df_직장인구.columns if "남성" in col and any(age in col for age in ["10", "20", "30"])]
OB_cols_male = [col for col in df_직장인구.columns if "남성" in col and any(age in col for age in ["40", "50", "60"])]
YB_cols_female = [col for col in df_직장인구.columns if "여성" in col and any(age in col for age in ["10", "20", "30"])]
OB_cols_female = [col for col in df_직장인구.columns if "여성" in col and any(age in col for age in ["40", "50", "60"])]

df_직장인구["남성_YB_직장_인구"] = df_직장인구[YB_cols_male].sum(axis=1)
df_직장인구["남성_OB_직장_인구"] = df_직장인구[OB_cols_male].sum(axis=1)
df_직장인구["여성_YB_직장_인구"] = df_직장인구[YB_cols_female].sum(axis=1)
df_직장인구["여성_OB_직장_인구"] = df_직장인구[OB_cols_female].sum(axis=1)
df_직장인구["YB_직장_인구"] = df_직장인구["남성_YB_직장_인구"] + df_직장인구["여성_YB_직장_인구"]
df_직장인구["OB_직장_인구"] = df_직장인구["남성_OB_직장_인구"] + df_직장인구["여성_OB_직장_인구"]

df_직장인구_grouped = df_직장인구.groupby("행정동_코드")[[
    "YB_직장_인구", "OB_직장_인구",
    "남성_YB_직장_인구", "남성_OB_직장_인구",
    "여성_YB_직장_인구", "여성_OB_직장_인구"
]].mean().reset_index()

# 6. 유동인구 (YB/OB, 성별 포함)
df_유동인구 = pd.read_csv(file_paths["길단위인구"], encoding='cp949')
df_유동인구 = df_유동인구[df_유동인구["기준_년분기_코드"].isin(quarters)]

YB_cols = [col for col in df_유동인구.columns if any(age in col for age in ["연령대_10", "연령대_20", "연령대_30"])]
OB_cols = [col for col in df_유동인구.columns if any(age in col for age in ["연령대_40", "연령대_50", "연령대_60"])]

df_유동인구["YB_유동인구"] = df_유동인구[YB_cols].sum(axis=1)
df_유동인구["OB_유동인구"] = df_유동인구[OB_cols].sum(axis=1)

df_유동인구_grouped = df_유동인구.groupby("행정동_코드")[[
    "총_유동인구_수", "남성_유동인구_수", "여성_유동인구_수", "YB_유동인구", "OB_유동인구"
]].mean().reset_index()

# 병합
data_frames = [
    df_상권변화,
    df_소득소비,
    df_아파트,
    df_음식점수,
    df_직장인구_grouped,
    df_유동인구_grouped
]

df_2024_분석용 = reduce(lambda left, right: pd.merge(left, right, on="행정동_코드", how="inner"), data_frames)

print(df_2024_분석용.head(10))

plt.figure()
plt.plot(df_2024_분석용[ "YB_직장_인구"])
plt.xlabel('기준 년분기 코드')
plt.ylabel('개업 점포 수')
plt.title(f'행정동 개업 점포 수 추이')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""-------------------------------------------------------------------------------------
2. class 나누기
-------------------------------------------------------------------------------------"""
# 1. KMeans, K=5

분석_변수 = [
    "월_평균_소득_금액", "음식_지출_총금액", "아파트_평균_시가", "YB_직장_인구",
    "OB_직장_인구", "남성_유동인구_수", "여성_유동인구_수"
]

X = df_2024_분석용[분석_변수].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
df_2024_분석용["cluster_k5"] = kmeans.fit_predict(X_scaled)

df_2024_분석용.to_csv("KMeans_분석결과.csv", index=False, encoding="utf-8-sig")

# 2. Hierachical

linked = linkage(X_scaled, method='ward')
df_2024_분석용["cluster_hier5"] = fcluster(linked, t=5, criterion='maxclust')

"""-------------------------------------------------------------------------------------
3. 지도 시각화화
-------------------------------------------------------------------------------------"""

# 1. KMeans
shp_path = "C:/Users/lluke/OneDrive/바탕 화면/SDC Data/서울시 상권분석서비스(영역-행정동).shp"
gdf_서울시 = gpd.read_file(shp_path)

csv_path = "C:/Users/lluke/OneDrive/바탕 화면/SDC Data/KMeans_분석결과.csv"
df_KMeans = pd.read_csv(csv_path)

gdf_서울시["ADSTRD_CD"] = gdf_서울시["ADSTRD_CD"].astype(str)
df_KMeans["행정동_코드"] = df_KMeans["행정동_코드"].astype(str)

gdf_지도_시각화_병합_K = gdf_서울시.rename(columns={"ADSTRD_CD": "행정동_코드"})
gdf_지도_시각화_K = gdf_지도_시각화_병합_K.merge(df_KMeans[["행정동_코드", "cluster_k5"]], on="행정동_코드", how="left")

fig, ax = plt.subplots(figsize=(12, 12))
gdf_지도_시각화_K.plot(column="cluster_k5", ax=ax, legend=True, cmap="Set3", edgecolor="black")
ax.set_title("서울시 행정동 K-Means cluster (k=5)", fontsize=15)
plt.axis("off")
plt.show()

# 2. Hierachical
csv_path_2 = "C:/Users/lluke/OneDrive/바탕 화면/SDC Data/hierachical_분석결과.csv"
df_Hier = pd.read_csv(csv_path_2)

df_Hier["행정동_코드"] = df_Hier["행정동_코드"].astype(str)

gdf_지도_시각화_병합_H = gdf_서울시.rename(columns={"ADSTRD_CD": "행정동_코드"})
gdf_지도_시각화_H = gdf_지도_시각화_병합_H.merge(df_Hier[["행정동_코드", "cluster_hier5"]], on="행정동_코드", how="left")

fig, ax = plt.subplots(figsize=(12, 12))
gdf_지도_시각화_H.plot(column="cluster_hier5", ax=ax, legend=True, cmap="Set3", edgecolor="black")
ax.set_title("서울시 행정동 hierachical cluster (k=5)", fontsize=15)
plt.axis("off")
plt.show()

