import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx


def preprocess(df, min_start_len, min_start_ratio):
    # for negative strand change y-axis vector direction
    df.loc[df['strand2'] == '-', ['start2+', 'end2+']] = df.loc[df['strand2'] == '-', ['end2+', 'start2+']].values
    df["id_ratio"] = df["id%"].str.rstrip("%").astype(float) / 100
    df = df[df.length1>=min_start_len]
    df = df[df.id_ratio>=min_start_ratio].reset_index(drop=True)
    df["seq_id"] = df.index
    df = df[['seq_id', 'start1', 'end1', 'start2+', 'end2+', 'strand2']]
    df.columns = ['seq_id', 'x1', 'x2', 'y1', 'y2', 'strand2']
    return df

# Функция для вычисления евклидова расстояния между двумя точками
def euclidean_distance(x1, y1, x2, y2):
    return np.linalg.norm([x2 - x1, y2 - y1])

def is_above_diagonal(x1, y1, x2, y2):
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    return ym > xm  # если выше диагонали

def plot_segments(df, ax, mask_merging=None, title=None):
    for i, row in df.iterrows():
        color = 'blue'
        if mask_merging is not None:
            color = 'red' if mask_merging[i] else 'blue'
        ax.plot([row['x1'], row['x2']], [row['y1'], row['y2']], color=color, lw=2)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True)

# Функция для объединения отрезков
def merge_segments(df, threshold=1000, plot=True):
    
    df = df.copy()
    merged_segments = []  # Список для объединённых отрезков
    merged_ids = set()  # Множество для отслеживания объединённых отрезков

    # Разделяем на 4 группы по strand2 и выше/ниже диагонали
    groups = {
        '+_above': [],
        '+_below': [],
        '-_above': [],
        '-_below': []
    }

    for idx, row in df.iterrows():
        if row['strand2'] == '+':
            if is_above_diagonal(row['x1'], row['y1'], row['x2'], row['y2']):
                groups['+_above'].append(idx)
            else:
                groups['+_below'].append(idx)
        elif row['strand2'] == '-':
            if is_above_diagonal(row['x1'], row['y1'], row['x2'], row['y2']):
                groups['-_above'].append(idx)
            else:
                groups['-_below'].append(idx)

    # Для каждой группы ищем объединение
    for group, indices in groups.items():
        subset = df.loc[indices]
        if len(subset) < 2:
            continue  # Если в группе меньше 2 отрезков, пропускаем

        # Убедимся, что все значения в столбцах x1 и y1 числовые
        subset = subset[pd.to_numeric(subset['x1'], errors='coerce').notnull()]
        subset = subset[pd.to_numeric(subset['y1'], errors='coerce').notnull()]

        # Сортируем отрезки по начальной точке
        subset = subset.sort_values(by=['x1', 'y1'], ascending=[True, True])

        # Создаём граф, где вершины — отрезки, а рёбра — это пара отрезков, которые можно объединить
        G = nx.Graph()

        # Добавляем все отрезки в граф
        for i in range(len(subset)):
            G.add_node(i, segment=subset.iloc[i])

        # Сравниваем все пары отрезков
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                dist = euclidean_distance(subset.iloc[i]['x2'], subset.iloc[i]['y2'], subset.iloc[j]['x1'], subset.iloc[j]['y1'])

                if dist < threshold:
                    G.add_edge(i, j)

        # Объединяем отрезки, связанные рёбрами в графе
        for component in nx.connected_components(G):
            if len(component) > 1:
                # Собираем все отрезки в компоненте
                min_seq_id = min(subset.iloc[list(component)]['seq_id'])  # Находим минимальный seq_id
                if subset.iloc[0]['strand2'] == '+':
                    merged_segment = {
                        'seq_id': min_seq_id,
                        'x1': subset.iloc[list(component)].min()['x1'],
                        'y1': subset.iloc[list(component)].min()['y1'],
                        'x2': subset.iloc[list(component)].max()['x2'],
                        'y2': subset.iloc[list(component)].max()['y2'],
                        'strand2': subset.iloc[0]['strand2']
                    }
                else:
                    merged_segment = {
                        'seq_id': min_seq_id,
                        'x1': subset.iloc[list(component)].min()['x1'],
                        'y1': subset.iloc[list(component)].max()['y1'],
                        'x2': subset.iloc[list(component)].max()['x2'],
                        'y2': subset.iloc[list(component)].min()['y2'],
                        'strand2': subset.iloc[0]['strand2']
                    }
                merged_segments.append(merged_segment)
                merged_ids.update(subset.iloc[list(component)].index)

    # Собираем DataFrame с объединёнными отрезками
    merged_df = pd.DataFrame(merged_segments)
    not_merged_df = df.drop(index=merged_ids)
    df_merged = pd.concat([merged_df, not_merged_df], ignore_index=True).sort_values(by='seq_id')
    mask_merging = df.index.isin(merged_ids)

    if plot==True:
        # Создаём два графика как subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # График 1: Отображаем оригинальные отрезки с маской объединённых
        plot_segments(df, ax1, mask_merging, title='Segments (red = to merge, blue = single)')
        
        # График 2: Отображаем объединённые отрезки
        plot_segments(df_merged, ax2)
        
        plt.tight_layout()
        plt.show()

    return df_merged

def symmetric(df3, plot=True):
    segments = set(((row.x1, row.y1), (row.x2, row.y2)) for _, row in df3.iterrows())
    non_symmetric_indices = []
    for idx, row in df3.iterrows():
        if row.strand2=='+':
            segment = ((row.x1, row.y1), (row.x2, row.y2))
            mirrored = ((row.y1, row.x1), (row.y2, row.x2))
            if mirrored not in segments:
                non_symmetric_indices.append(idx)
        else:
            segment = ((row.x1, row.y1), (row.x2, row.y2))
            mirrored = ((row.y2, row.x2), (row.y1, row.x1))
            if mirrored not in segments:
                non_symmetric_indices.append(idx)
    df4 = df3.drop(non_symmetric_indices).reset_index(drop=True)
    if plot==True:
        # Создаём два графика как subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # График 1: Отображаем оригинальные отрезки с красными для ненаправленных
        for i, row in df3.iterrows():
            color = 'red' if i in non_symmetric_indices else 'blue'
            ax1.plot([row['x1'], row['x2']], [row['y1'], row['y2']], color=color, lw=2)
            #ax1.text((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2, str(row['seq_id']), fontsize=8)
        
        ax1.invert_yaxis()
        ax1.set_aspect("equal")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Original Segments (red = non-symmetric, blue = symmetric)')
        ax1.grid(True)
        
        # График 2: Отображаем объединённые отрезки
        for i, row in df4.iterrows():
            ax2.plot([row['x1'], row['x2']], [row['y1'], row['y2']], color='blue', lw=2)
            #ax2.text((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2, str(row['seq_id']), fontsize=8)
        
        ax2.invert_yaxis()
        ax2.set_aspect("equal")
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Symmetric segments')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    return df4

def find_negative_sites(df):    

    # 1. Отфильтровываем "-" отрезки
    df5 = df[df.strand2 == '-'].copy()
    
    # === ОБРАБОТКА ОСИ X ===
    # Получаем интервалы и сортируем
    intervals_x = df5[['x1', 'x2']].values.tolist()
    intervals_x.sort(key=lambda x: x[0])
    
    # Объединяем пересекающиеся интервалы по X
    merged_x = []
    for start, end in intervals_x:
        if not merged_x or merged_x[-1][1] < start:
            merged_x.append([start, end])
        else:
            merged_x[-1][1] = max(merged_x[-1][1], end)
    
    # df5_x: объединённые интервалы по X
    df5_x = pd.DataFrame(merged_x, columns=['x1_m', 'x2_m']).reset_index().rename(columns={'index': 'proj_id_x'})
    #print(df5_x)
    # Назначаем каждому отрезку из df5 границы объединённого интервала по X
    def assign_projection_x(row):
        x1, x2 = row['x1'], row['x2']
        for _, proj in df5_x.iterrows():
            if not (x2 < proj.x1_m or x1 > proj.x2_m):  # пересечение
                return pd.Series([proj.proj_id_x, proj.x1_m, proj.x2_m])
        return pd.Series([None, None, None])
    
    df5[['proj_id_x', 'x1_m', 'x2_m']] = df5.apply(assign_projection_x, axis=1)
    
    # === ОБРАБОТКА ОСИ Y ===
    # Получаем интервалы и сортируем
    intervals_y = df5.apply(lambda row: sorted([row['y1'], row['y2']]), axis=1).tolist()
    intervals_y.sort(key=lambda x: x[0])
    
    # Объединяем пересекающиеся интервалы по Y
    merged_y = []
    for start, end in intervals_y:
        if not merged_y or merged_y[-1][1] < start:
            merged_y.append([start, end])
        else:
            merged_y[-1][1] = max(merged_y[-1][1], end)
    
    # df5_y: объединённые интервалы по Y
    df5_y = pd.DataFrame(merged_y, columns=['y1_m', 'y2_m']).reset_index().rename(columns={'index': 'proj_id_y'})
    #print(df5_y)
    # Назначаем каждому отрезку из df5 границы объединённого интервала по Y
    def assign_projection_y(row):
        y1, y2 = sorted([row['y1'], row['y2']])
        for _, proj in df5_y.iterrows():
            if not (y2 < proj.y1_m or y1 > proj.y2_m):
                return pd.Series([proj.proj_id_y, proj.y1_m, proj.y2_m])
        return pd.Series([None, None, None])
        
    df5[['proj_id_y', 'y1_m', 'y2_m']] = df5.apply(assign_projection_y, axis=1)
    
    def assign_projection_x(row):
        x1, x2 = row['x1'], row['x2']
        for _, proj in df5_x.iterrows():
            if not (x2 < proj.x1_m or x1 > proj.x2_m):
                return pd.Series([proj.proj_id_x, proj.x1_m, proj.x2_m])
        return pd.Series([None, None, None])
    
    
    def assign_projection_y(row):
        y1, y2 = sorted([row['y1'], row['y2']])
        for _, proj in df5_y.iterrows():
            if not (y2 < proj.y1_m or y1 > proj.y2_m):
                return pd.Series([proj.proj_id_y, proj.y1_m, proj.y2_m])
        return pd.Series([None, None, None])
    
    df5[['proj_id_x', 'x1_m', 'x2_m']] = df5.apply(assign_projection_x, axis=1)
    df5[['proj_id_y', 'y1_m', 'y2_m']] = df5.apply(assign_projection_y, axis=1)

    df5_sites = df5_x
    df5_sites.columns = ['proj_id', 'x1_m', 'x2_m']
    df5_sites['y1_m'] = df5_sites['x1_m']
    df5_sites['y2_m'] = df5_sites['x2_m']
    
    return df5.sort_index(), df5_sites

# def find_negative_set(df6, df5_sites, plot=False):
    
#     # 1. Построим граф
#     G = nx.Graph()
#     edges = df6[['proj_id_x', 'proj_id_y']].dropna().astype(int).drop_duplicates().values.tolist()
#     edges = [edge for edge in edges if edge[0] != edge[1]]
#     G.add_edges_from(edges)
    
#     # 2. Проверка на двудольность и раскраска
#     def is_bipartite_and_color_graph(G):
#         if nx.is_bipartite(G):
#             coloring = nx.bipartite.color(G)
#             return True, coloring
#         else:
#             return False, None
    
#     # 3. Проверка графа
#     is_bipartite, coloring = is_bipartite_and_color_graph(G)
    
#     # 4. Визуализация
#     if is_bipartite:
#         # Если граф двудольный, рисуем с раскраской
#         if plot:
#             plt.figure(figsize=(10, 7))
#             pos = nx.spring_layout(G)  # Расположение вершин
#             nx.draw(G, pos, with_labels=True, node_color=[coloring[node] for node in G.nodes()],
#                     node_size=500, font_size=10, font_weight='bold', edge_color='gray', cmap=plt.cm.RdYlBu)
#             plt.title("Двудольный граф с раскраской")
#             plt.show()
    
#         # 5. Рассчитываем веса вершин (node_weight) с использованием x1_m и x2_m
#         node_weights = {
#             node: abs(df6.loc[df6['proj_id_x'] == node, 'x2_m'].values[0] - df6.loc[df6['proj_id_x'] == node, 'x1_m'].values[0])
#             for node in G.nodes()
#         }
    
#         # 6. Суммируем веса вершин для каждой из долей
#         set_0 = set([node for node, color in coloring.items() if color == 0])
#         set_1 = set([node for node, color in coloring.items() if color == 1])
    
#         weight_0 = sum(node_weights[node] for node in set_0)
#         weight_1 = sum(node_weights[node] for node in set_1)
    
#         # 7. Выбираем долю с меньшим суммарным весом
#         if weight_0 < weight_1:
#             smaller_set = set_0
#         else:
#             smaller_set = set_1
    
#         # 8. Выводим номера вершин с меньшим суммарным весом
#         #print(f"Номера вершин с меньшей суммарной длиной (веса): {smaller_set}")
#     else:
#         print("Граф не является двудольным, раскраска невозможна.")

#     negative_sites = df5_sites[df5_sites.proj_id.isin(smaller_set)]

#     return smaller_set, negative_sites

def find_negative_set(df6, df5_sites, plot=False):
    # 1. Построим граф
    G = nx.Graph()
    edges = df6[['proj_id_x', 'proj_id_y']].dropna().astype(int).drop_duplicates().values.tolist()
    edges = [edge for edge in edges if edge[0] != edge[1]]
    G.add_edges_from(edges)

    # 2. Проверка на двудольность и раскраска
    def is_bipartite_and_color_graph(G):
        if nx.is_bipartite(G):
            coloring = nx.bipartite.color(G)
            return True, coloring
        else:
            return False, None

    is_bipartite, coloring = is_bipartite_and_color_graph(G)

    if is_bipartite:
        # 3. Рассчитываем веса вершин
        node_weights = {
            node: abs(
                df6.loc[df6['proj_id_x'] == node, 'x2_m'].values[0] -
                df6.loc[df6['proj_id_x'] == node, 'x1_m'].values[0]
            )
            for node in G.nodes()
        }

        # 4. Разделение на доли
        set_0 = set([node for node, color in coloring.items() if color == 0])
        set_1 = set([node for node, color in coloring.items() if color == 1])

        weight_0 = sum(node_weights[node] for node in set_0)
        weight_1 = sum(node_weights[node] for node in set_1)

        smaller_set = set_0 if weight_0 < weight_1 else set_1
        negative_sites = df5_sites[df5_sites.proj_id.isin(smaller_set)]

        return {
            "smaller_set": smaller_set,
            "negative_sites": negative_sites,
            "graph": G,
            "is_bipartite": True,
            "coloring": coloring,
            "node_weights": node_weights
        }
    else:
        return {
            "smaller_set": set(),
            "negative_sites": pd.DataFrame(),
            "graph": G,
            "is_bipartite": False,
            "coloring": None,
            "node_weights": {}
        }

def reflect_segments_with_visuals(df, negative_dict):
    df_reflected_x = df.copy()
    reflected_x_idx = set()

    df_segments = negative_dict['negative_sites']

    # Этап 1: отражение по оси X
    for i in range(df_segments.shape[0]):
        x1_1, x2_1 = df_segments.iloc[i][["x1_m", "x2_m"]]
        mirror_x = (x1_1 + x2_1) / 2
        #print('границы:', x1_1, x2_1)
        #print('mirror_x:', mirror_x)

        df_within = df_reflected_x[(df_reflected_x['x1'] >= x1_1) & (df_reflected_x['x2'] <= x2_1)].copy()
        reflected_x_idx.update(df_within.index)

        df_within['x1'] = mirror_x - (df_within['x1'] - mirror_x)
        df_within['x2'] = mirror_x - (df_within['x2'] - mirror_x)

        df_non = df_reflected_x.drop(index=df_within.index)
        df_reflected_x = pd.concat([df_non, df_within])

    mask = (df_reflected_x['x1'] > df_reflected_x['x2']) & (df_reflected_x['y1'] > df_reflected_x['y2'])
    df_reflected_x.loc[mask, ['x1', 'x2']] = df_reflected_x.loc[mask, ['x2', 'x1']].values
    df_reflected_x.loc[mask, ['y1', 'y2']] = df_reflected_x.loc[mask, ['y2', 'y1']].values

    # Этап 2: отражение по оси Y
    df_reflected_xy = df_reflected_x.copy()
    reflected_y_idx = set()

    for i in range(df_segments.shape[0]):
        x1_1, x2_1 = df_segments.iloc[i][["x1_m", "x2_m"]]
        mirror_y = (x1_1 + x2_1) / 2
        #print('границы:', x1_1, x2_1)
        #print('mirror_y:', mirror_y)

        df_within = df_reflected_xy[(df_reflected_xy['y1'] >= x1_1) & (df_reflected_xy['y2'] <= x2_1)].copy()
        reflected_y_idx.update(df_within.index)

        df_within['y1'] = mirror_y - (df_within['y1'] - mirror_y)
        df_within['y2'] = mirror_y - (df_within['y2'] - mirror_y)

        df_non = df_reflected_xy.drop(index=df_within.index)
        df_reflected_xy = pd.concat([df_non, df_within])

    # Финальный swap
    mask = (df_reflected_xy['x1'] > df_reflected_xy['x2']) & (df_reflected_xy['y1'] > df_reflected_xy['y2'])
    df_reflected_xy.loc[mask, ['x1', 'x2']] = df_reflected_xy.loc[mask, ['x2', 'x1']].values
    df_reflected_xy.loc[mask, ['y1', 'y2']] = df_reflected_xy.loc[mask, ['y2', 'y1']].values

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    
    def plot_segments(ax, df_all, df_green_idx, title):
        for i, row in df_all.iterrows():
            color = 'green' if i in df_green_idx else 'blue'
            ax.plot([row["x1"], row["x2"]], [row["y1"], row["y2"]], color=color, linewidth=2)
    
        # Границы и оси отображения
        for _, seg in df_segments.iterrows():
            x1_m, x2_m = seg["x1_m"], seg["x2_m"]
            mirror = (x1_m + x2_m) / 2
    
            # Границы зоны
            ax.axvline(x1_m, color='red', linestyle='--', linewidth=1)
            ax.axvline(x2_m, color='red', linestyle='--', linewidth=1)
            ax.axhline(x1_m, color='red', linestyle='--', linewidth=1)
            ax.axhline(x2_m, color='red', linestyle='--', linewidth=1)
    
            # Оси отражения
            ax.axvline(mirror, color='red', linestyle=':', linewidth=1.5)
            ax.axhline(mirror, color='red', linestyle=':', linewidth=1.5)
    
        ax.set_title(title)
        ax.set_xlabel("x1 - x2")
        ax.set_ylabel("y1 - y2")
        ax.set_aspect("equal")
        ax.invert_yaxis()
    
    # Граф
    if negative_dict["is_bipartite"]:
        pos = nx.spring_layout(negative_dict["graph"])
        nx.draw(negative_dict["graph"], pos,
                node_color=[negative_dict["coloring"][n] for n in negative_dict["graph"].nodes()],
                with_labels=True, cmap=plt.cm.RdYlBu, ax=axs[0, 0])
        axs[0, 0].set_title("")
    
    # Отображение сегментов
    plot_segments(axs[0, 1], df, set(), "Original")
    plot_segments(axs[1, 0], df_reflected_x, reflected_x_idx, "Reflected in X (green)")
    plot_segments(axs[1, 1], df_reflected_xy, reflected_y_idx, "Reflected in X & Y (green)")
    
    plt.tight_layout()
    plt.show()
    
    return df_reflected_xy