import pandas as pd
from sklearn.cluster import DBSCAN
import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Polygon


def apply_dbscan(df, col1, col2, dbscan_eps):
    coords = df[[col1, col2]].values
    clustering = DBSCAN(eps=dbscan_eps, min_samples=1).fit(coords)
    return clustering.labels_

def find_split_points_dbscan(df, dbscan_eps):

    df2 = df.copy()
    df2["cluster_x1"] = apply_dbscan(df2, "x1", "x1", dbscan_eps)
    df2["cluster_y1"] = apply_dbscan(df2, "y1", "y1", dbscan_eps)
    df2["cluster_x2"] = apply_dbscan(df2, "x2", "x2", dbscan_eps)
    df2["cluster_y2"] = apply_dbscan(df2, "y2", "y2", dbscan_eps)

    
    split_points_x1 = df2.groupby(by='cluster_x1').filter(lambda x: 
                                                          len(x) >= 2).groupby('cluster_x1')['x1'].max().tolist()
    split_points_x2 = df2.groupby(by='cluster_x2').filter(lambda x: 
                                                          len(x) >= 2).groupby('cluster_x2')['x2'].min().tolist()
    
    split_points = sorted(split_points_x1 + split_points_x2)

    return split_points

# import numpy as np
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import itertools

class DSeq:
    def __init__(self, seq_id, x1, x2, y1, y2):
        self.seq_id = seq_id
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def projection(self, axis):
        """Returns the projection of the segment on the specified axis."""
        return (self.x1, self.x2) if axis == 'x' else (self.y1, self.y2)

    def subsegment_in_band(self, axis, band_s, band_e):
        """Возвращает подотрезок, лежащий в полосе [band_s, band_e]."""
        if axis == 'x':
            new_s, new_e, s_fixed, e_fixed = self.x1, self.x2, self.y1, self.y2
        else:
            new_s, new_e, s_fixed, e_fixed = self.y1, self.y2, self.x1, self.x2

        new_s = max(new_s, band_s)
        new_e = min(new_e, band_e)

        if new_s >= new_e:
            return None  # Нет пересечения
        
        ra = (self.y2 - self.y1) / (self.x2 - self.x1)
        new_fixed_s = s_fixed + (new_s - (self.x1 if axis == 'x' else self.y1)) * ra
        new_fixed_e = e_fixed - ((self.x2 if axis == 'x' else self.y2) - new_e) * ra

        return DSeq(self.seq_id, *(int(new_s), int(new_e), int(new_fixed_s), int(new_fixed_e)) if axis == 'x' else (int(new_fixed_s), int(new_fixed_e), int(new_s), int(new_e)))

    def split_by(self, axis, value):
        """Разделяет отрезок по оси axis на два, если точка value внутри отрезка."""
        if axis == 'x':
            if value <= self.x1 or value >= self.x2:
                return [self]
        else:
            if value <= self.y1 or value >= self.y2:
                return [self]

        ra = (self.y2 - self.y1) / (self.x2 - self.x1)
        new_fixed = self.y1 + (value - self.x1) * ra if axis == 'x' else self.x1 + (value - self.y1) / ra

        return [
            DSeq(self.seq_id, *(self.x1, value-1, self.y1, int(new_fixed)-1) if axis == 'x' else (self.x1, int(new_fixed)-1, self.y1, value-1)),
            DSeq(self.seq_id, *(value, self.x2, int(new_fixed), self.y2) if axis == 'x' else (int(new_fixed), self.x2, value, self.y2))
        ]

    def __repr__(self):
        return f"DSeq(seq_id={self.seq_id}, x1={self.x1}, x2={self.x2}, y1={self.y1}, y2={self.y2})"

def split_segments_by_values(df, min_length=1000, split_points_x=None, split_points_y=None):
    """
    Разрезает отрезки по заданной сетке split_points_x (вертикальные линии) и split_points_y (горизонтальные линии),
    фильтруя отрезки длиной меньше min_length.
    """
    segments = [DSeq(row.seq_id, row.x1, row.x2, row.y1, row.y2) for _, row in df.iterrows()]
    
    if split_points_x is None:
        split_points_x = sorted(set(df["x1"]).union(df["x2"]))
    if split_points_y is None:
        split_points_y = sorted(set(df["y1"]).union(df["y2"]))

    # Разрезаем по вертикальным линиям (split_points_x)
    new_segments = []
    for seg in segments:
        temp_segs = [seg]
        for x in split_points_x:
            temp_segs = [sub_seg for seg in temp_segs for sub_seg in seg.split_by('x', x)]
        new_segments.extend(temp_segs)

    # Разрезаем по горизонтальным линиям (split_points_y)
    final_segments = []
    for seg in new_segments:
        temp_segs = [seg]
        for y in split_points_y:
            temp_segs = [sub_seg for seg in temp_segs for sub_seg in seg.split_by('y', y)]
        final_segments.extend(temp_segs)

    # Фильтруем отрезки по минимальной длине
    filtered_segments = [
        seg for seg in final_segments 
        if (abs(seg.x2 - seg.x1) >= min_length) and (abs(seg.y2 - seg.y1) >= min_length)
    ]

    # Преобразуем обратно в DataFrame
    df_split = pd.DataFrame([vars(seg) for seg in filtered_segments])
    return df_split

def intersect_ratio(seg1, seg2, axis='x'):
    """Вычисляет, насколько два отрезка пересекаются на заданной оси (x или y)."""
    if axis == 'x':
        start1, end1 = seg1.x1, seg1.x2
        start2, end2 = seg2.x1, seg2.x2
    elif axis == 'y':
        start1, end1 = seg1.y1, seg1.y2
        start2, end2 = seg2.y1, seg2.y2
    else:
        return 0

    # Находим пересекающуюся длину
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start < overlap_end:
        overlap_length = overlap_end - overlap_start
    else:
        overlap_length = 0
    
    # Находим объединение отрезков
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_length = union_end - union_start
    
    if union_length == 0:
        return 0  # Защита от деления на 0 (если отрезки совпадают, но длина объединения равна 0)
    
    # Возвращаем отношение пересечения к объединению
    return overlap_length / union_length

def create_graph(segments, threshold=0.8):
    """Создает граф, где отрезки соединяются, если они пересекаются на >= threshold по X или Y."""
    G = nx.Graph()
    
    # Добавляем отрезки как узлы
    for i, seg in enumerate(segments):
        G.add_node(i, segment=seg)
    
    # Соединяем отрезки, которые пересекаются
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments):
            if i >= j:
                continue
            if intersect_ratio(seg1, seg2, axis='x') >= threshold or intersect_ratio(seg1, seg2, axis='y') >= threshold:
                G.add_edge(i, j)
    
    return G


def assign_clusters(G):
    """Разбивает граф на компоненты связности и присваивает кластеры."""
    components = list(nx.connected_components(G))
    cluster_mapping = {}
    
    for cluster_id, component in enumerate(components):
        for node in component:
            cluster_mapping[node] = cluster_id
    
    return cluster_mapping

def create_graph_from_segments(segments, threshold=0.8):
    """
    Создает граф из сегментов, где сегменты пересекаются, если они достаточно близки по координатам.
    
    :param segments: список объектов сегментов (DSeq или подобных объектов, с аттрибутами x1, x2, y1, y2).
    :param threshold: порог для перекрытия, чтобы считать два сегмента пересекающимися.
    :return: граф с ребрами, которые связывают пересекающиеся сегменты.
    """
    # Создаем пустой граф
    G = nx.Graph()

    # Добавляем сегменты как узлы в граф
    for i, segment in enumerate(segments):
        G.add_node(i, seq_id=segment.seq_id, x1=segment.x1, x2=segment.x2, y1=segment.y1, y2=segment.y2)

    # Функция для проверки пересечения двух сегментов
    def intersect(segment1, segment2):
        # Проверяем, есть ли пересечение между сегментами
        return not (segment1.x2 < segment2.x1 or segment2.x2 < segment1.x1)

    # Добавляем ребра между пересекающимися сегментами
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments):
            if i >= j:
                continue  # Не проверяем сегменты самими с собой
            if intersect(seg1, seg2):
                G.add_edge(i, j)

    return G


def is_overlap(interval1, interval2):
    x1_1, x2_1 = interval1
    x1_2, x2_2 = interval2
    return not (x2_1 <= x1_2 or x2_2 <= x1_1)
    
def loop(df2, split_points, min_len_after_split, graph_sim_ratio):
    df2 = split_segments_by_values(df2, split_points_x = split_points, split_points_y = split_points, min_length = min_len_after_split)
    #print('after split 2:', df2.shape[0])
    
    #plot_segments_rand(df2, "Random Colored Segments", figsize=(4, 4))
    
    segments = [DSeq(row.seq_id, row.x1, row.x2, row.y1, row.y2) for _, row in df2.iterrows()]
    G = create_graph(segments, threshold=graph_sim_ratio)
    
    cluster_mapping = assign_clusters(G)
    df2['cluster'] = df2.index.map(cluster_mapping)
    df2['seq_id'] = df2.index
    
    # print(f'Number of clusters: {len(set(df2.cluster))}')
    
    all_clustered_segments = []
    for cluster_id in df2['cluster'].unique():
        df_cluster = df2[df2['cluster'] == cluster_id].copy()
        
        segments = [DSeq(row.seq_id, row.x1, row.x2, row.y1, row.y2) for _, row in df_cluster.iterrows()]
        
        G = create_graph_from_segments(segments, threshold=graph_sim_ratio)
        components = list(nx.connected_components(G))
        subcluster_list = [-1] * df_cluster.shape[0]
        for subcluster_id, component in enumerate(components):
            for node in component:
                subcluster_list[node] = subcluster_id
        df_cluster['subcluster'] = subcluster_list
        all_clustered_segments.append(df_cluster)
    
    #!!!!!
    df2 = pd.concat(all_clustered_segments)
    df3 = df2.groupby(['cluster', 'subcluster'], as_index=False).agg({'x1': 'mean', 'x2': 'mean'})
    #df3 = df2.groupby(['cluster', 'subcluster'], as_index=False).agg({'x1': 'max', 'x2': 'min'})
    df3 = df3[df3['cluster'].map(df3['cluster'].value_counts()) > 1]
    
    # plot
    #plot_segments_with_clusters(df2, cluster_mapping)
    # plot
    #plot_median_segments(df3)
    
    all_clustered_segments = []
    for cluster_id in df2['cluster'].unique():
        df_cluster = df2[df2['cluster'] == cluster_id].copy()
        
        segments = [DSeq(row.seq_id, row.x1, row.x2, row.y1, row.y2) for _, row in df_cluster.iterrows()]
        
        G = create_graph_from_segments(segments, threshold=graph_sim_ratio)
        components = list(nx.connected_components(G))
        subcluster_list = [-1] * df_cluster.shape[0]
        for subcluster_id, component in enumerate(components):
            for node in component:
                subcluster_list[node] = subcluster_id
        df_cluster['subcluster'] = subcluster_list
        all_clustered_segments.append(df_cluster)
    
    #!!!!!
    df2 = pd.concat(all_clustered_segments)
    df3 = df2.groupby(['cluster', 'subcluster'], as_index=False).agg({'x1': 'mean', 'x2': 'mean'})
    #df3 = df2.groupby(['cluster', 'subcluster'], as_index=False).agg({'x1': 'max', 'x2': 'min'})
    df3 = df3[df3['cluster'].map(df3['cluster'].value_counts()) > 1]
    
    
    print(f'Number of clusters: {len(set(df3.cluster))}')
    
    df4 = df3[['cluster', 'x1', 'x2']].copy()
    df4 = df4.sort_values(by=['x1', 'x2']).reset_index(drop=True)
    
    overlapping_pairs = []
    
    G = nx.Graph()
    
    for i in range(1, len(df4)):  # начинаем с 1, чтобы сравнивать только соседние
        row1 = df4.iloc[i-1]
        row2 = df4.iloc[i]
        if is_overlap((row1['x1'], row1['x2']), (row2['x1'], row2['x2'])):
            overlapping_pairs.append((row1['cluster'], row2['cluster']))
            G.add_edge(row1['cluster'], row2['cluster'])
    
    
    connected_components = list(nx.connected_components(G))
    #print("Connected components:", connected_components)
    # plot
    #plot_segments_with_clusters(df2, cluster_mapping)
    # plot
    #plot_median_segments(df3)
    return df2, df4, connected_components

def plot_necklace(ax, df, info_text=None, title=None, colors = None, s_min = None, e_max = None):

    random.seed(0)
    if colors == None:
        num_components = df["cluster"].nunique()
        palette = sns.color_palette("Spectral", num_components)
        random.shuffle(palette)
        colors = {comp: palette[i] for i, comp in enumerate(sorted(df["cluster"].unique()))}
    
    ax.set_title(title)
    
    for _, row in df.iterrows():
        comp = row["cluster"]
        color = colors[comp]
    
        x_center = (row["x1"] + row["x2"]) / 2
        y_center = x_center  
    
        width = (row["x2"] - row["x1"]) * 1.37  
        height = width * 0.2 * 4.0
    
        diamond = np.array([
            [0, height / 2],
            [width / 2, 0],
            [0, -height / 2],
            [-width / 2, 0]
        ])
        
        angle = np.radians(45)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        rotated_diamond = diamond @ rotation_matrix.T  
        rotated_diamond[:, 0] += x_center  
        rotated_diamond[:, 1] += y_center  
    
        polygon = Polygon(rotated_diamond, edgecolor=color, facecolor=color, alpha=0.8)
        ax.add_patch(polygon)
    
    ax.set_xlim(s_min-1000, e_max+1000)
    ax.set_ylim(s_min-1000, e_max+1000)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("x1 - x2")
    ax.set_ylabel("y1 - y2")

    if info_text: ax.text(
        0.95, 0.95, info_text,
        transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.25, edgecolor='black'))
    
    return colors

def plot_segemnts(ax, df, df_reverse_segments = None, title='', x1=None, y1=None, x2=None, y2=None, show_box=False, show_bands=False, 
                 band_width=10000, x1_min=None, x2_max=None, reverse_segments = None):

    ax.set_title(title)

    for _, row in df.iterrows():
        ax.plot([row["x1"], row["x2"]], [row["y1"], row["y2"]], color='b', linewidth=2)

    if df_reverse_segments is not None and not df_reverse_segments.empty:
        for _, row in df_reverse_segments.iterrows():
            ax.plot([row["x1"], row["x2"]], [row["x1"], row["x2"]], color='red', linewidth=2)

    if x1_min==None:
        x1_min = df["x1"].min()
    if x2_max==None:
        x2_max = df["x2"].max()
    ax.set_xlim(x1_min-1000, x2_max+1000)
    ax.set_ylim(x1_min-1000, x2_max+1000)
    
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("x1 - x2")
    ax.set_ylabel("y1 - y2")

def make_fig(df, df_reflected, df4, df_reverse_regions, res_dir, speciex1, title=None):

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    x_min, x_max = df["x1"].min(), df["x2"].max()
    red_squares = [[x_min-1, x_max+1]]

    df7 = df[(df.x1 >= x_min-100) & (df.x2 <= x_max+100) & (df.y1 >= x_min-100) & (df.y2 <= x_max+100)]
    df6 = df4[(df4.x1 >= df7.x1.min()) & (df4.x2 <= df7.x2.max())]
    df7_v2 = df_reflected[(df_reflected.x1 >= x_min-100) & (df_reflected.x2 <= x_max+100) & (df_reflected.y1 >= x_min-100) & (df_reflected.y2 <= x_max+100)]

    
    
    info_text = ''
    plot_segemnts(axes[0], df7, title='')
    plot_segemnts(axes[1], df7_v2, df_reverse_segments=df_reverse_regions, title='')
    colors = plot_necklace(axes[2], df6, info_text=info_text, s_min = df7.x1.min(), e_max = df7.x2.max())
        
    plt.tight_layout()
    #plt.savefig(res_dir+speciex1+'.png', dpi=300, bbox_inches='tight')
    plt.show()